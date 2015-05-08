#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <string>
#include <cstdio>
#include <fstream>
#include <sstream>

#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ml/ml.hpp>

#include "../volume.h"
#include "../volume_visualizer.h"
#include "../tessellation_generator.h"
#include "../features.h"

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// THIS ENTIRE BLOCK IS JUST TO DUMP USEFUL STATISTICS ABOUT THE RESULTS OF THE TRAINING PROCESS

struct CloudInfo {
    pcl::PointXYZ pose, velocity;
    float thetaDeg; // person orientation. 180° = looking INTO camrea
    float sensorDistance;
    float phi; // angle between optical axis of camera and person
    size_t numPoints;
};

// For looking up which column in a sample's feature vector belongs to which feature of which volume of which tessellation.
struct FeatureVectorLookupEntry {
    size_t tessellationIndex;
    size_t volumeInTessellationIndex;
    size_t overallVolumeIndex;
    size_t featureIndex;
};

struct HistogramEntry {
    size_t numberOfTimesUsed;
    float accumulatedQuality;
};

// To detect which features are being used the most often
struct FeatureHistogramEntry : public HistogramEntry {
    size_t featureIndex;
};

// To detect which volumes (voxels) are being used the most often
struct VolumeHistogramEntry  : public HistogramEntry {
    size_t overallVolumeIndex;
    size_t tessellationIndex;
    size_t volumeInTessellationIndex;
};

struct HistogramEntryComparatorByNumberOfTimesUsed {
    bool operator() (const HistogramEntry& lhs, const HistogramEntry& rhs) const {
       return lhs.numberOfTimesUsed > rhs.numberOfTimesUsed;
    }
};

struct HistogramEntryComparatorByAccumulatedQuality {
    bool operator() (const HistogramEntry& lhs, const HistogramEntry& rhs) const {
       return lhs.accumulatedQuality > rhs.accumulatedQuality;
    }
};

// For sorting splits (decision stumps) of Adaboost classifier
struct SplitComparatorByQuality {
    bool operator() ( CvDTreeSplit* const lhs, CvDTreeSplit* const rhs) const {
       return lhs->quality > rhs->quality;
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Global variables
std::list<Tessellation> g_tessellations;
std::vector<string> g_featureNames;
std::vector<Volume> g_overallVolumeLookup;
ros::Publisher g_pointCloudPublisher;
Volume g_parentVolume;
size_t g_numPositiveSamples, g_numNegativeSamples, g_overallVolumeCount, g_minPoints;
bool g_dumpFeatures;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double testClassifier(const CvBoost& adaboost, const cv::Mat& labels, const cv::Mat& features, const cv::Mat& sampleIdx, const cv::Mat& missingDataMask,
    const std::vector<std::string>& cloudFilenames, const std::vector<CloudInfo>& cloudInfos, const std::string& csvFilename)
{
    size_t correctlyClassified = 0, totalClassified = 0;

    std::ofstream csvFile( csvFilename.c_str() );
    csvFile << "classifier,mode,identifier,prediction,correct,position_x,position_y,velocity_x,velocity_y,orientation,phi,num_points" << std::endl;

    ROS_INFO("rows: labels=%d, features=%d, sampleIdx=%d", labels.rows, features.rows, sampleIdx.rows);

    ROS_ASSERT(labels.rows == features.rows && labels.rows == sampleIdx.rows);
    for(size_t sample = 0; sample < labels.rows; sample++) {
        unsigned char sampleIsActive = sampleIdx.at<unsigned char>(sample);

        if(sampleIsActive) {
            int label = labels.at<int>(sample);
            cv::Mat featureVector = features.row(sample);
            cv::Mat missingDataVector = missingDataMask.row(sample);

            float predictedLabel = adaboost.predict(featureVector, missingDataVector);
            bool correct = int(predictedLabel) == label;
            if(correct) correctlyClassified++;

            totalClassified++;

            const std::string& cloudFilename = cloudFilenames[sample];
            std::string identifier = boost::filesystem::path(cloudFilename).stem().string();
            boost::replace_all(identifier, "_cloud", "");

            csvFile << "adaboost,test," << identifier << "," << ((int)predictedLabel+1) << "," << (correct ? 1 : 0);
            
            const CloudInfo& cloudInfo = cloudInfos[sample];
            csvFile << ",";
            csvFile << cloudInfo.pose.x << "," << cloudInfo.pose.y << "," << cloudInfo.velocity.x << "," << cloudInfo.velocity.y << "," << cloudInfo.thetaDeg << ",";
            csvFile << cloudInfo.phi << "," << cloudInfo.numPoints;
            csvFile << std::endl;
        }
    }

    return double(correctlyClassified) / totalClassified;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void calculateFeaturesOnDataset(int fold, bool useValidationSet, cv::Mat& labels, cv::Mat& features, cv::Mat& sampleIdx, cv::Mat& missingDataMask,
    std::vector<FeatureVectorLookupEntry>& featureVectorLookup, std::vector<std::string>& cloudFilenames, std::vector<CloudInfo>& cloudInfos)
{
    PointCloud::Ptr personCloud(new PointCloud);

    // Count total number of volumes (voxels) over all tessellations
    g_overallVolumeCount = 0;
    foreach(Tessellation tessellation, g_tessellations) {
        foreach(Volume volume, tessellation.getVolumes()) {
            g_overallVolumeCount++;
        }
    }

    // Count number of features
    FeatureCalculator featureCalculator;
    g_featureNames = featureCalculator.getFeatureNames();
    const size_t featureCount = g_featureNames.size();
    const size_t featureVectorSize = g_overallVolumeCount * featureCount;

    ROS_INFO_STREAM_ONCE("Total count of tessellation volumes (voxels):  " << g_overallVolumeCount);
    ROS_INFO_STREAM_ONCE("Number of different features: " << featureCount);
    ROS_INFO_STREAM_ONCE("--> Each person cloud will have a feature vector of dimension " << featureVectorSize);


    //
    // Load training data set and calculate features
    //
    ROS_INFO_STREAM("Initializing " << (useValidationSet ? "validation" : "training") << " set for fold " << fold << "...");

    std::stringstream trainFilename;
    trainFilename << ros::package::getPath(ROS_PACKAGE_NAME) << "/data/fold" << std::setw(3) << std::setfill('0') << fold << "/" << (useValidationSet ? "val" : "train") << ".txt";
    std::ifstream listFile(trainFilename.str().c_str());

    string cloudFilename;
    int label;
    size_t numClouds = 0;
    set<int> labelSet;

    // Skip comments at beginning of file
    std::string commentString;
    const size_t numCommentLinesAtBeginning = 2;
    for(size_t i = 0; i < numCommentLinesAtBeginning; i++) std::getline(listFile, commentString);

    // See how many cloud files we have
    while (listFile >> cloudFilename >> label)
    {
        numClouds++;
    }

    // Go back to start of list file
    listFile.clear();
    listFile.seekg(0, std::ios::beg);
    for(size_t i = 0; i < numCommentLinesAtBeginning; i++) std::getline(listFile, commentString);

    // Initialize OpenCV matrices
    labels.create(numClouds, 1, CV_32SC1);
    labels.setTo(cv::Scalar(99999));

    features.create(numClouds, featureVectorSize, CV_32FC1);
    features.setTo(cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
    
    sampleIdx.create(numClouds, 1, CV_8UC1);
    sampleIdx.setTo(cv::Scalar(0));

    missingDataMask.create(numClouds, featureVectorSize, CV_8UC1);
    missingDataMask.setTo(cv::Scalar(1));

    // Clear other fields
    g_numNegativeSamples = g_numPositiveSamples = 0;
    cloudFilenames.clear();
    cloudInfos.clear();

    // Create map for looking up which column in the feature vector belongs to which tesellation + voxel + feature
    g_overallVolumeLookup.clear();
    g_overallVolumeLookup.reserve(g_overallVolumeCount);
    featureVectorLookup.clear();
    featureVectorLookup.reserve(featureVectorSize);
    bool featureVectorLookupInitialized = false;


    ROS_INFO_STREAM("Loading " << numClouds << " clouds to calculate features...");

    // Now start calculating features
    size_t numSkippedLowQualityClouds = 0;
    ros::WallRate rate(10);
    int cloudCounter = 0;
    while (listFile >> cloudFilename >> label && ros::ok())
    {
        // Show progress
        if(cloudCounter % (numClouds / 10) == 0) {
            ROS_INFO("%d %% of feature computations done...", int(cloudCounter / (float)numClouds * 100.0f + 0.5f));
        }

        // Load pose file
        std::string poseFilename = cloudFilename;
        boost::replace_all(poseFilename, "_cloud.pcd", "_pose.txt");
        std::ifstream poseFile(poseFilename.c_str());
        if(poseFile.fail()) {
            ROS_FATAL("Couldn't read pose file %s\n", poseFilename.c_str());
            continue;
        }

        std::string fieldName, NaNString;

        float NaN = std::numeric_limits<float>::quiet_NaN();
        CloudInfo cloudInfo;
        cloudInfo.pose.x = NaN; cloudInfo.pose.y = NaN; cloudInfo.pose.z = 0;
        cloudInfo.velocity.x = NaN; cloudInfo.velocity.y = NaN; cloudInfo.velocity.z = 0;
        cloudInfo.thetaDeg = NaN; cloudInfo.sensorDistance = NaN;

        // The fail() / clear() mechanism is needed because istream cannot handle NaN values
        poseFile >> fieldName >> cloudInfo.pose.x;            if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "position_x");
        poseFile >> fieldName >> cloudInfo.pose.y;            if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "position_y");
        poseFile >> fieldName >> cloudInfo.velocity.x;        if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "velocity_x");
        poseFile >> fieldName >> cloudInfo.velocity.y;        if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "velocity_y");
        poseFile >> fieldName >> cloudInfo.thetaDeg;          if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "theta_deg");
        poseFile >> fieldName >> cloudInfo.sensorDistance;    if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "sensor_distance");

        cloudInfo.phi = atan2(cloudInfo.pose.y, cloudInfo.pose.x) * 180.0 / M_PI;
        
        // Load PCD file
        if(pcl::io::loadPCDFile<PointType>(cloudFilename, *personCloud) == -1)
        {
            ROS_FATAL("Couldn't read file %s\n", cloudFilename.c_str());
            continue;
        }
        else 
        {
            personCloud->header.frame_id = "extracted_cloud_frame";
            personCloud->header.stamp = ros::Time::now().toNSec() / 1000;
            g_pointCloudPublisher.publish(personCloud);
        }

        // Number of points is the last meta-data item we need to decide about goodness of cloud
        cloudInfo.numPoints = personCloud->points.size();
        
        // Skip low-quality clouds during training stage
        if(!useValidationSet) {
            if(cloudInfo.sensorDistance < 1.1 || abs(cloudInfo.phi) > 30.0 || cloudInfo.numPoints < 3500) {
                ROS_DEBUG("Skipping cloud %s which does not meet goodness criteria.\nDist: %.1f  Phi: %.1f  #Points: %d", cloudFilename.c_str(), cloudInfo.sensorDistance, abs(cloudInfo.phi), (int)cloudInfo.numPoints);
                numSkippedLowQualityClouds++;
                continue;
            }      
        }

        // Everything good! Now store meta data and name of input cloud (for dump of results later on)
        cloudInfos.push_back(cloudInfo);
        cloudFilenames.push_back(cloudFilename);

        // Store label
        labels.at<int>(cloudCounter) = label;
        labelSet.insert(label);
        
        if(label > 0) g_numPositiveSamples++;
        else g_numNegativeSamples++; 

        // Calculate features
        std::vector<double> fullFeatureVectorForCloud;

        size_t featureColumn = 0, t = 0, overallVolumeIndex = 0;
        foreach(Tessellation tessellation, g_tessellations)  // for each tessellation...
        {
            #pragma omp parallel for schedule(dynamic) ordered
            for(size_t v = 0; v < tessellation.getVolumes().size(); v++)  // for each volume in that tessellation...
            {
                // Get points inside volume
                std::vector<int> indicesInsideVolume;
                const Volume& volume = tessellation.getVolumes()[v];
                volume.getPointsInsideVolume(*personCloud, PointCloud::Ptr(), &indicesInsideVolume);

                // Calculate features (if sufficient points inside volume)
                std::vector<double> volumeFeatureVector;
                const size_t MIN_POINT_COUNT = g_minPoints;
                if(indicesInsideVolume.size() >= MIN_POINT_COUNT) {
                    featureCalculator.calculateFeatures(g_parentVolume, *personCloud, indicesInsideVolume,
                        featureCalculator.maskAllFeaturesActive(), volumeFeatureVector); 
                }
                else volumeFeatureVector = std::vector<double>(featureCount, std::numeric_limits<double>::quiet_NaN());

                // Copy feature values into right spot of sample's overall feature vector
                ROS_ASSERT(volumeFeatureVector.size() == featureCount);
                    
                #pragma omp ordered
                #pragma omp critical
                {
                    for(size_t f = 0; f < volumeFeatureVector.size(); f++)  // for each feature...
                    {
                        features.at<float>(cloudCounter, featureColumn) = volumeFeatureVector[f];
                        missingDataMask.at<unsigned char>(cloudCounter, featureColumn) = !std::isfinite(volumeFeatureVector[f]) ? 1 : 0;

                        if(!featureVectorLookupInitialized) {
                            FeatureVectorLookupEntry entry;
                            entry.tessellationIndex = t;
                            entry.volumeInTessellationIndex = v;
                            entry.overallVolumeIndex = overallVolumeIndex;
                            entry.featureIndex = f;
                            featureVectorLookup.push_back(entry);
                        }
                        featureColumn++;
                    }

                    if(!featureVectorLookupInitialized) {
                        g_overallVolumeLookup.push_back(volume);
                    }

                    overallVolumeIndex++;
                }
            }
            t++;
        }

        featureVectorLookupInitialized = true;
        ROS_ASSERT(featureVectorLookup.size() == featureVectorSize);
        ROS_ASSERT(g_overallVolumeLookup.size() == g_overallVolumeCount);

        // Mark sample as active
        sampleIdx.at<unsigned char>(cloudCounter) = 1;

        // Prepare for next sample
        cloudCounter++;
        ros::spinOnce();
        rate.sleep();
    }


    // Dump feature values + labels into file for off-line analysis
    if(g_dumpFeatures) {
        ROS_INFO("Dumping training data into file for optional off-line analysis...");
        std::string prefix( std::string("top_down_") + std::string(useValidationSet ? "val" : "train") );
        std::ofstream featureFile( (prefix + "_features.txt").c_str() );
        std::ofstream missingFile( (prefix + "_missing.txt").c_str() );
        std::ofstream labelFile( (prefix + "_labels.txt").c_str() );

        featureFile << features;
        missingFile << missingDataMask;
        labelFile << labels;
    }

    std::stringstream ss;
    foreach(int label, labelSet) {
        ss << label << " ";
    }
    ROS_INFO_STREAM("Labels in input data: " << ss.str());
    ROS_INFO_STREAM("Feature computation complete on " << (useValidationSet ? "validation" : "training") << " set!");

    if(!useValidationSet) {
        ROS_INFO_STREAM("Number of clouds skipped in training phase due to low quality: " << numSkippedLowQualityClouds << " ("
                        << std::fixed << std::setprecision(2) << (numSkippedLowQualityClouds / (float)numClouds * 100.0f) << "%)");
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void generatePermutationsOfAspectRatios(std::vector<pcl::PointXYZ>& voxelAspectRatios)
{
    std::vector<pcl::PointXYZ> newAspectRatios;

    foreach(pcl::PointXYZ aspectRatio, voxelAspectRatios) {
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.x, aspectRatio.z, aspectRatio.y) );
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.z, aspectRatio.x, aspectRatio.y) );
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.z, aspectRatio.y, aspectRatio.x) );
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.y, aspectRatio.x, aspectRatio.z) );
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.y, aspectRatio.z, aspectRatio.x) );       
    }

    voxelAspectRatios.insert(voxelAspectRatios.end(), newAspectRatios.begin(), newAspectRatios.end());
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void writeVolume(cv::FileStorage& fileStorage, const Volume& volume) {
    const pcl::PointXYZ& minCoords = volume.getMinCoordinates();
    const pcl::PointXYZ& maxCoords = volume.getMaxCoordinates();

    fileStorage << "{:";
        fileStorage << "minCoords" << "{:" << "x" << minCoords.x << "y" << minCoords.y << "z" << minCoords.z << "}"; 
        fileStorage << "maxCoords" << "{:" << "x" << maxCoords.x << "y" << maxCoords.y << "z" << maxCoords.z << "}"; 
    fileStorage << "}";
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "top_down_classifier_training");
    ros::NodeHandle nodeHandle("");
    ros::NodeHandle privateHandle("~");

    bool showTessellations, showBestTessellation, interactive;
    privateHandle.param<bool>("show_tessellations", showTessellations, true);
    privateHandle.param<bool>("show_best_tessellation", showBestTessellation, true);
    privateHandle.param<bool>("interactive", interactive, false);
    privateHandle.param<bool>("dump_features", g_dumpFeatures, false);

    int initialFold, numFolds;
    privateHandle.param<int>("fold", initialFold, 0);
    privateHandle.param<int>("num_folds", numFolds, 1); // set to e.g. 5 if there are five splitX folders in "data" (cross-validation)
    
    g_minPoints = 4; 

    omp_set_num_threads(5);


    //
    // Load test point cloud
    //

    PointCloud::Ptr personCloud(new PointCloud);
    string filename = ros::package::getPath(ROS_PACKAGE_NAME) + "/data/test_cloud.pcd";
    ROS_INFO_STREAM("Loading " << filename << "...");
    
    if(pcl::io::loadPCDFile<PointType>(filename, *personCloud) == -1)
    {
        ROS_ERROR("Couldn't read file %s\n", filename.c_str());
        return (-1);
    }

    // Create point cloud publisher
    g_pointCloudPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("cloud", 1, true);
    personCloud->header.frame_id = "extracted_cloud_frame";
    
    // Instantiate visualizer
    VolumeVisualizer visualizer;

    // Generate bounding (parent) volume
    ROS_INFO_STREAM("Getting parent volume...");

    pcl::PointXYZ minCoords(-0.3, -0.3, 0), maxCoords(0.3, 0.3, 1.8);
    g_parentVolume = Volume(minCoords, maxCoords); // = Volume::fromCloudBBox(*personCloud);

    pcl::PointXYZ parentVolumeSize = g_parentVolume.getSize();
    ROS_INFO("Parent volume has size %.2g %.2g %.2g", parentVolumeSize.x, parentVolumeSize.y, parentVolumeSize.z);


    //
    // Initialize tessellation generator
    //

    double minVoxelSize, regularTessellationSize;
    bool overlapEnabled, regularTessellationOnly;

    privateHandle.param<bool>("overlap", overlapEnabled, true);
    privateHandle.param<bool>("regular_tessellation_only", regularTessellationOnly, false);
    privateHandle.param<double>("regular_tessellation_size", regularTessellationSize, 0.1);
    privateHandle.param<double>("min_voxel_size", minVoxelSize, 0.1);
    
    
    std::vector<pcl::PointXYZ> voxelAspectRatios; 

    voxelAspectRatios.push_back( pcl::PointXYZ(1, 1, 1) );
    
    if(!regularTessellationOnly) {
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 2.5) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 5.0) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 4.0) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 6.0) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 8.0) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 10.0) );

        voxelAspectRatios.push_back( pcl::PointXYZ(0.1, 0.1, 1.8) );
        voxelAspectRatios.push_back( pcl::PointXYZ(0.2, 0.2, 1.8) );
        voxelAspectRatios.push_back( pcl::PointXYZ(0.3, 0.3, 1.8) );

        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 1.25) );

        voxelAspectRatios.push_back( pcl::PointXYZ(2, 2, 2) );
        voxelAspectRatios.push_back( pcl::PointXYZ(3, 3, 3) );
        voxelAspectRatios.push_back( pcl::PointXYZ(4, 4, 4) );

        voxelAspectRatios.push_back( pcl::PointXYZ(1, 1, 2) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1, 1, 3) );
        voxelAspectRatios.push_back( pcl::PointXYZ(2, 2, 3) );
        voxelAspectRatios.push_back( pcl::PointXYZ(2, 3, 3) ); // new
        voxelAspectRatios.push_back( pcl::PointXYZ(4, 4, 3) );
        voxelAspectRatios.push_back( pcl::PointXYZ(4, 4, 2) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1, 4, 4) );
    }

    generatePermutationsOfAspectRatios(voxelAspectRatios);


    std::vector<float> voxelSizeIncrements;
    if(!regularTessellationOnly) {
        voxelSizeIncrements.push_back(0.1);
        voxelSizeIncrements.push_back(0.2);
        voxelSizeIncrements.push_back(0.3);
        voxelSizeIncrements.push_back(0.4);
        voxelSizeIncrements.push_back(0.5);
        voxelSizeIncrements.push_back(0.6);
        voxelSizeIncrements.push_back(0.7);
        voxelSizeIncrements.push_back(0.8);
        voxelSizeIncrements.push_back(0.9);
        voxelSizeIncrements.push_back(1.0);
        voxelSizeIncrements.push_back(1.1);
    }
    else {
        voxelSizeIncrements.push_back(regularTessellationSize);
    } 

    TessellationGenerator tessellationGenerator(g_parentVolume, voxelAspectRatios, voxelSizeIncrements, minVoxelSize, overlapEnabled);


    //
    // Generate tessellations
    //

    ROS_INFO_STREAM("Beginning to generate tessellations (overlap " << (overlapEnabled ? "enabled" : "disabled") << ")..." );
    tessellationGenerator.generateTessellations(g_tessellations);
    ROS_INFO_STREAM("Finished generating tessellations! Got " << g_tessellations.size() << " in total!");


    //
    // Visualize tessellations
    //

    ros::WallRate rate(1);
    ROS_INFO_STREAM("Visualizing tessellations...");
    
    // TEST: Use only 1st tessellation in regular tessellation-only mode
    if(regularTessellationOnly) {
        std::list<Tessellation>::iterator it;
        while(g_tessellations.size() > (overlapEnabled ? 2 : 1)) {
            it = g_tessellations.begin();
            std::advance(it, g_tessellations.size() - 1);
            g_tessellations.erase(it);
        }
    }

    size_t tessellationIndex = 0;
    std::list<Tessellation>::const_iterator tessellationIt = g_tessellations.begin();

    while(showTessellations && ros::ok()) {
        personCloud->header.stamp = ros::Time::now().toNSec() / 1000;
        g_pointCloudPublisher.publish(personCloud);

        visualizer.clear();
        visualizer.visualize(g_parentVolume, 0, "Parent Volume", 0xffff00, 3.0f);

        if(tessellationIndex == 0 && interactive) {
            visualizer.publish();
            ros::spinOnce();
            ROS_INFO("Press ENTER to show first tessellation...");
            getchar();
        }

        if(!g_tessellations.empty()) {
            for(int i = 0; i < (overlapEnabled ? 2 : 1); i++) { // normal + overlapping tessellation
                ROS_INFO_STREAM("Showing tessellation #" << tessellationIndex << (overlapEnabled && i & 1 ? " (overlapping)" : "" ) );
                const std::vector<Volume>& volumes = tessellationIt->getVolumes();
                for(size_t volumeInTessellationIndex = 0; volumeInTessellationIndex < volumes.size(); volumeInTessellationIndex++) {
                   visualizer.visualize(volumes[volumeInTessellationIndex], 1 + volumeInTessellationIndex, tessellationIt->isOverlapping() ? "Tessellation (Overlapping)" : "Tessellation"); 
                }

                if(tessellationIndex < g_tessellations.size() - 1) {
                    tessellationIt++;
                    tessellationIndex++;
                }
            }
        }

        visualizer.publish();
        visualizer.publish();
        ros::spinOnce();
        ros::spinOnce();
        ros::spinOnce();
        visualizer.publish();
        

        rate.sleep();
        
        if(interactive) {
            ROS_INFO("Press ENTER to show next tessellation...");
            getchar();
        }

        if(tessellationIndex < g_tessellations.size() - 1) {
            tessellationIndex++;
            tessellationIt++;
        }
        else {
            break;
            //tessellationIndex = 0;
            //tessellationIt = g_tessellations.begin();
        }
    }

    // Done with showing tessellations, show only parent volume
    visualizer.clear();
    visualizer.visualize(g_parentVolume, 0, "Parent Volume", 0xffff00, 3.0f);
    visualizer.publish();
    ros::spinOnce();


    //
    // Perform training across all selected training/test set folds
    //
    std::vector<double> testAccuracyPerFold;
    std::vector< cv::Ptr<CvBoost> > learnedClassifiers;

    double bestTestAccuracySoFar = 0;
    cv::Ptr<CvBoost> bestClassifierSoFar;

    std::stringstream ss;
    std::vector<FeatureVectorLookupEntry> featureVectorLookup;

    // Iterate over folds
    ros::WallTime startOfFold; ros::WallDuration durationOfPreviousFold;

    for(int fold = initialFold; fold < initialFold + numFolds; fold++) {
        ROS_INFO_STREAM("");
        ROS_INFO_STREAM("### STARTING TO PROCESS FOLD " << fold-initialFold+1 << " (=" << fold << ") " << " OF " << numFolds << "! ###");

        startOfFold = ros::WallTime::now();

        if(fold-initialFold > 0) {
            ROS_INFO_STREAM("Estimated time to completion of all folds: " << (durationOfPreviousFold.toSec() * (numFolds - (fold - initialFold))) / 60.0 << " minutes...");
        }

        //
        // Calculate features on training set
        //

        ROS_INFO_STREAM("");
        ROS_INFO_STREAM("=== STARTING TRAINING PHASE: FEATURE COMPUTATIONS! ===");

        cv::Mat labels, features, sampleIdx, missingDataMask;
        std::vector<std::string> cloudFilenames;
        std::vector<CloudInfo> cloudInfos;
        calculateFeaturesOnDataset(fold, false, labels, features, sampleIdx, missingDataMask, featureVectorLookup, cloudFilenames, cloudInfos);
        

        //
        // Train Adaboost classifier
        //

        ROS_INFO_STREAM("");
        ROS_INFO_STREAM("=== STARTING TRAINING PHASE: LEARN ADABOOST CLASSIFIER! ===");

        // Output results of previous folds, while user is waiting for training to complete
        if(!testAccuracyPerFold.empty()) {
            ss.str("");
            double averageTestAccuracy = 0;
            foreach(double testAccuracy, testAccuracyPerFold) {
                ss << std::fixed << std::setprecision(2) << testAccuracy * 100.0 << "% ";
                averageTestAccuracy += testAccuracy;
            }
            averageTestAccuracy /= (double) testAccuracyPerFold.size();
            ROS_INFO_STREAM("Test accuracies of previous fold(s): " << ss.str());
            ROS_INFO("Average test accuracy after %d fold(s): %.2f%%, best: %.2f%%", fold-initialFold, averageTestAccuracy * 100.0, bestTestAccuracySoFar * 100.0);
        }
        ROS_INFO_STREAM("Now in fold " << fold-initialFold+1 << " of " << numFolds << "...");

        int weakClassifierCount; privateHandle.param<int>("weak_classifiers", weakClassifierCount, 100);

        CvBoostParams params;
        params.boost_type=CvBoost::REAL; //DISCRETE;
        params.weight_trim_rate = 0.95; // default: 0.95 (to speed up computation time); set to 0 to turn off this feature
        params.weak_count = weakClassifierCount; 
        params.use_surrogates = false; // 'true' required to handle missing measurements (???) <-- will create lots of additional tree splits
        params.max_depth = 1;
        params.truncate_pruned_tree = true;
        privateHandle.getParam("weak_count", params.weak_count);

        ROS_INFO_STREAM("Training Adaboost classifier... this may take a while!");
        ROS_INFO_STREAM("Number of weak classifiers: " << params.weak_count << ", weight trim rate: " << params.weight_trim_rate);

        cv::Ptr<CvBoost> adaboost(new CvBoost);
        if(!adaboost->train(features, CV_ROW_SAMPLE, labels, cv::Mat(), sampleIdx, cv::Mat(), missingDataMask, params)) {
            ROS_ERROR("Training of Adaboost classifier failed for unknown reason!");
        }

        learnedClassifiers.push_back(adaboost);


        //
        // Test Adaboost classifier
        //

        ROS_INFO_STREAM("");
        ROS_INFO_STREAM("=== STARTING TESTING PHASE: TESTING CLASSIFIER ON TRAINING SET! ===");
        ss.str(""); ss << "top_down_train_results_fold" << fold << ".csv";
        double trainAccuracy = testClassifier(*adaboost, labels, features, sampleIdx, missingDataMask, cloudFilenames, cloudInfos, ss.str());
        ROS_INFO("Accuracy on training set: %.2f%%", trainAccuracy * 100.0);

        // Load test set
        ROS_INFO_STREAM("");
        ROS_INFO_STREAM("=== STARTING TESTING PHASE: TESTING CLASSIFIER ON TEST SET! ===");
        
        calculateFeaturesOnDataset(fold, true, labels, features, sampleIdx, missingDataMask, featureVectorLookup, cloudFilenames, cloudInfos);  // true = use validation set

        ss.str(""); ss << "top_down_test_results_fold" << fold << ".csv";
        double testAccuracy = testClassifier(*adaboost, labels, features, sampleIdx, missingDataMask, cloudFilenames, cloudInfos, ss.str());
        ROS_INFO("Accuracy on test set: %.2f%%", testAccuracy * 100.0);

        if(testAccuracy > bestTestAccuracySoFar) {
            bestTestAccuracySoFar = testAccuracy;
            bestClassifierSoFar = adaboost;
        }
        testAccuracyPerFold.push_back(testAccuracy);


        // Update timing info
        durationOfPreviousFold = ros::WallTime::now() - startOfFold;

    } // end of current run (fold)


    //
    // Show test accuracies across all folds
    //
    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("### FINISHED PROCESSING ALL TRAINING/TEST FOLDS! ###");

    ss.str("");
    double averageTestAccuracy = 0;
    foreach(double testAccuracy, testAccuracyPerFold) {
        ss << std::fixed << std::setprecision(2) << testAccuracy * 100.0 << "% ";
        averageTestAccuracy += testAccuracy;
    }
    averageTestAccuracy /= (double) testAccuracyPerFold.size();
    ROS_INFO_STREAM("Test accuracies across all folds: " << ss.str());
    ROS_INFO("Average test accuracy: %.2f%%, best: %.2f%%", averageTestAccuracy * 100.0, bestTestAccuracySoFar * 100.0);


    //
    // Save best learned Adaboost classifier
    //

    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("=== SAVING ADABOOST CLASSIFIER WHICH PERFORMED BEST IN TESTING! ==="); 

    cv::FileStorage fileStorage("top_down_classifier.yaml", cv::FileStorage::WRITE);

    // Names of all features
    fileStorage << "feature_names" << "[";
    for(size_t i = 0; i < g_featureNames.size(); i++) {
        fileStorage << g_featureNames[i];
    }
    fileStorage << "]";

    // Size of training set
    //fileStorage << "training_set_num_clouds" << (int) cv::sum(sampleIdx).val[0];
    //fileStorage << "training_set_num_positive_samples" << (int)g_numPositiveSamples;
    //fileStorage << "training_set_num_negative_samples" << (int)g_numNegativeSamples;
    fileStorage << "num_tessellations" << (int)g_tessellations.size();
    fileStorage << "num_overall_volumes" << (int)g_overallVolumeCount;
    fileStorage << "feature_vector_length" << (int)featureVectorLookup.size();
    fileStorage << "min_points" << (int)g_minPoints;

    // The parent volume used for training
    fileStorage << "parent_volume";
    writeVolume(fileStorage, g_parentVolume);

    // The voxels of all tessellations
    fileStorage << "tessellation_volumes" << "[";
    foreach(const Tessellation& tessellation, g_tessellations) {
        foreach(const Volume& volume, tessellation.getVolumes()) {
            writeVolume(fileStorage, volume);
        }
    }
    fileStorage << "]";

    // Entries of the full feature vector per person cloud
    fileStorage << "feature_vector_entries" << "[";
    for(size_t f = 0; f < featureVectorLookup.size(); f++) {
        const FeatureVectorLookupEntry& entry = featureVectorLookup[f];
        fileStorage << "{:" << "tessellationIndex" << (int)entry.tessellationIndex 
                            << "volumeInTessellationIndex" << (int)entry.volumeInTessellationIndex
                            << "overallVolumeIndex" << (int)entry.overallVolumeIndex
                            << "featureIndex" << (int)entry.featureIndex 
                    << "}";
    }
    fileStorage << "]";

    // Write the classifier itself
    bestClassifierSoFar->write(*fileStorage, "classifier");

    // Close file
    fileStorage.release();


    //
    // Examine resulting weak classifiers of best learned classifier
    //

    // Extract weak classifiers from Adaboost classifier (NOTE: this will change in OpenCV 3)
    std::vector<CvBoostTree*> weakClassifiers;
    CvSeq* sequenceOfWeakClassifiers = bestClassifierSoFar->get_weak_predictors();
    CvSeqReader reader; cvStartReadSeq(sequenceOfWeakClassifiers, &reader);

    for(size_t i = 0; i < sequenceOfWeakClassifiers->total; ++i)
    {
        CvBoostTree* tree;
        CV_READ_SEQ_ELEM(tree, reader);
        weakClassifiers.push_back(tree);
    }

    ROS_INFO_STREAM("Got " << weakClassifiers.size() << " decision trees from best learned classifier!");
    std::vector<CvDTreeSplit*> splits;
    
    foreach(CvBoostTree* tree, weakClassifiers) {
        const CvDTreeNode* node = tree->get_root();
        ROS_ASSERT(node != NULL);

        if(node->left) { // if there is no left child, there is no split
            CvDTreeSplit* split = node->split;
            while(split != NULL) {
                splits.push_back(split);
                split = split->next;
            }
        }
    }

    ROS_INFO_STREAM("Extracted " << splits.size() << " tree splits from best learned classifier!");
    std::stable_sort(splits.begin(), splits.end(), SplitComparatorByQuality());


    // Initialize histogram of most frequently used features and volumes (voxels)
    std::vector<FeatureHistogramEntry> featureHistogram;    
    for(size_t f = 0; f < g_featureNames.size(); f++) {
        FeatureHistogramEntry entry;
        entry.featureIndex = f;
        entry.numberOfTimesUsed = 0;
        entry.accumulatedQuality = 0;
        featureHistogram.push_back(entry);
    }

    std::vector<VolumeHistogramEntry> volumeHistogram;
    for(size_t v = 0; v < g_overallVolumeCount; v++) {
        VolumeHistogramEntry entry;
        entry.overallVolumeIndex = v;
        entry.numberOfTimesUsed = 0;
        entry.accumulatedQuality = 0;
        volumeHistogram.push_back(entry);
    }

    // Build histogram of most frequently used features and volumes (voxels)
    const size_t NUM_SPLITS_TO_SHOW = 30;
    ss.str("");
    for(size_t s = 0; s < splits.size(); s++) {
        const CvDTreeSplit& split = *splits[s];

        int varIdx = split.var_idx; // this is the column in the feature vector
        float quality = split.quality;
        size_t featureIndex = featureVectorLookup[varIdx].featureIndex;
        size_t overallVolumeIndex = featureVectorLookup[varIdx].overallVolumeIndex;
        size_t tessellationIndex = featureVectorLookup[varIdx].tessellationIndex;
        size_t volumeInTessellationIndex = featureVectorLookup[varIdx].volumeInTessellationIndex;

        if(s < NUM_SPLITS_TO_SHOW) {
            ss << "- Feature vector column #" << std::setw(6) << std::setfill('0') << varIdx << ": " << g_featureNames[featureIndex] << " = " << quality << std::endl;
        }

        featureHistogram[featureIndex].numberOfTimesUsed++;
        featureHistogram[featureIndex].accumulatedQuality += quality;
        volumeHistogram[overallVolumeIndex].numberOfTimesUsed++;
        volumeHistogram[overallVolumeIndex].tessellationIndex = tessellationIndex;
        volumeHistogram[overallVolumeIndex].volumeInTessellationIndex = volumeInTessellationIndex;
        volumeHistogram[overallVolumeIndex].accumulatedQuality += quality;
    }

    ROS_INFO_STREAM("First " << NUM_SPLITS_TO_SHOW << " splits ordered by descending quality: " << std::endl << ss.str());
    std::stable_sort(featureHistogram.begin(), featureHistogram.end(), HistogramEntryComparatorByNumberOfTimesUsed());
    std::stable_sort(volumeHistogram.begin(),  volumeHistogram.end(),  HistogramEntryComparatorByNumberOfTimesUsed());

    // Display most frequently used features and volumes (voxels)
    ss.str("");
    foreach(FeatureHistogramEntry& entry, featureHistogram) {
        ss << "- " << g_featureNames[entry.featureIndex] << " (" << entry.numberOfTimesUsed << "x)" << std::endl;
    }
    ROS_INFO_STREAM("Most frequently used features: \n" << ss.str());

    /*
    std::stable_sort(featureHistogram.begin(), featureHistogram.end(), HistogramEntryComparatorByAccumulatedQuality());
    ss.str("");
    foreach(FeatureHistogramEntry& entry, featureHistogram) {
        ss << "- " << g_featureNames[entry.featureIndex] << " (" << entry.accumulatedQuality << ")" << std::endl;
    }
    ROS_INFO_STREAM("Best features by accumulated quality: \n" << ss.str());
    */

    size_t volumeCount = 0;
    ss.str("");
    foreach(VolumeHistogramEntry& entry, volumeHistogram) {
        if(entry.numberOfTimesUsed >= 1) {
            ss << "- Volume " << entry.overallVolumeIndex << ", which is sub-volume " << entry.volumeInTessellationIndex << " of tessellation " << entry.tessellationIndex << " (" << entry.numberOfTimesUsed << "x)" << std::endl;
        }

        if(++volumeCount > 25) {
            ss << "..." << std::endl;
            break;
        }
    }
    ROS_INFO_STREAM("Most frequently occurring voxels (only those which occur more than 1x): \n" << ss.str());
    

    // Visualize best tessellation
    visualizer.clear();
    visualizer.visualize(g_parentVolume, 0, "Parent Volume", 0xffff00, 3.0f);

    personCloud->header.stamp = ros::Time::now().toNSec() / 1000;
    g_pointCloudPublisher.publish(personCloud); // show test cloud again

    rate = ros::WallRate(20);
    volumeCount = 0;
    foreach(VolumeHistogramEntry& entry, volumeHistogram) {
        if(!ros::ok() || !showBestTessellation) break;
        if(entry.numberOfTimesUsed < 1) break;

        const Volume& volume = g_overallVolumeLookup[entry.overallVolumeIndex];        

        visualizer.visualize(volume, 1 + entry.overallVolumeIndex, tessellationIt->isOverlapping() ? "Tessellation (Overlapping)" : "Tessellation"); 
    
        //rate.sleep();

        //if(++volumeCount > 25) {
        //    break;
        //}

        ++volumeCount;
        
        //ROS_INFO("Press ENTER to show next volume...");
        //getchar();
    }

    ROS_INFO_STREAM("Visualizing " << volumeCount << " learned tessellation volumes of best classifier...");
    visualizer.publish();
    ros::spinOnce();



    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("=== FINISHED! ===");
    
    // Quit
    if(!ros::ok()) {
        std::cout << std::endl << "Termination requested by user!" << std::endl;
        return 3;
    }

    return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
