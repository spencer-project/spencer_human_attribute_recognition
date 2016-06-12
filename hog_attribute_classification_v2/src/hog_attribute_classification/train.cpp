/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2014-2015, Timm Linder, Social Robotics Lab, University of Freiburg
*  All rights reserved.
*  
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*  
*  * Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*  * Neither the name of the copyright holder nor the names of its contributors
*    may be used to endorse or promote products derived from this software
*    without specific prior written permission.
*  
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
*  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
*  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
*  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <ros/ros.h>
#include <ros/package.h>

#include <string>
#include <cstdio>
#include <fstream>
#include <sstream>

#include <omp.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ml/ml.hpp>

#include "3rd_party/cnpy/cnpy.h"

using namespace std;

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>
#define foreach BOOST_FOREACH

std::string g_modality;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// THIS ENTIRE BLOCK IS JUST TO DUMP USEFUL STATISTICS ABOUT THE RESULTS OF THE TRAINING PROCESS
struct Vector3d {
    float x, y, z;
};

struct SampleInfo {
    Vector3d pose, velocity;
    float thetaDeg; // person orientation. 180° = looking INTO camrea
    float sensorDistance;
    float phi; // angle between optical axis of camera and person
    size_t numPoints;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Global variables
size_t g_numPositiveSamples, g_numNegativeSamples;
bool g_dumpFeatures, g_applyMasks, g_visualization;
std::string g_category, g_numpyExportFolder;
int g_winSizeX, g_winSizeY;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double testClassifier(const CvSVM& classifier, const cv::Mat& labels, const cv::Mat& features, const cv::Mat& sampleIdx, const cv::Mat& missingDataMask,
    const std::vector<std::string>& sampleFilenames, const std::vector<SampleInfo>& sampleInfos, const std::string& csvFilename)
{
    size_t correctlyClassified = 0, totalClassified = 0;

    std::ofstream csvFile( csvFilename.c_str() );
    csvFile << "classifier,mode,identifier,prediction,correct,position_x,position_y,velocity_x,velocity_y,orientation,phi,num_points,score" << std::endl;

    ROS_INFO("rows: labels=%d, features=%d, sampleIdx=%d", labels.rows, features.rows, sampleIdx.rows);

    ROS_ASSERT(labels.rows == features.rows && labels.rows == sampleIdx.rows);
    for(size_t sample = 0; sample < labels.rows; sample++) {
        unsigned char sampleIsActive = sampleIdx.at<unsigned char>(sample);

        if(sampleIsActive) {
            int label = labels.at<int>(sample);
            cv::Mat featureVector = features.row(sample);
            cv::Mat missingDataVector = missingDataMask.row(sample);

            // IMPORTANT: Comment out this first predict() call when doing speed benchmarks!!!
            ROS_WARN_ONCE("Calling predict() twice for each sample to obtain raw value of decision function. Uncomment this before benchmarking runtime performance!");
            float decisionFunctionValue = classifier.predict(featureVector, true);

            int predictedLabel = (int) classifier.predict(featureVector, false);
            bool correct = predictedLabel == label;
            if(correct) correctlyClassified++;

            totalClassified++;

            const std::string& sampleFilename = sampleFilenames[sample];
            std::string identifier = boost::filesystem::path(sampleFilename).stem().string();
            boost::replace_all(identifier, "_cloud", "");

            csvFile << "svm,test," << identifier << "," << (predictedLabel+1) << "," << (correct ? 1 : 0);
            
            const SampleInfo& sampleInfo = sampleInfos[sample];
            csvFile << ",";
            csvFile << sampleInfo.pose.x << "," << sampleInfo.pose.y << "," << sampleInfo.velocity.x << "," << sampleInfo.velocity.y << "," << sampleInfo.thetaDeg << ",";
            csvFile << sampleInfo.phi << "," << sampleInfo.numPoints << ",";
            csvFile << decisionFunctionValue;
            csvFile << std::endl;
        }
    }

    return double(correctlyClassified) / totalClassified;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool loadSampleInfo(std::string& poseFilename, SampleInfo& sampleInfo)
{
    std::ifstream poseFile(poseFilename.c_str());
    if(poseFile.fail()) {
        return false;
    }

    // initialize fields
    float NaN = std::numeric_limits<float>::quiet_NaN();
    sampleInfo.numPoints = 0;
    sampleInfo.pose.x = NaN; sampleInfo.pose.y = NaN; sampleInfo.pose.z = 0;
    sampleInfo.velocity.x = NaN; sampleInfo.velocity.y = NaN; sampleInfo.velocity.z = 0;
    sampleInfo.thetaDeg = NaN; sampleInfo.sensorDistance = NaN;

    // read the input file
    std::string line;
    while(std::getline(poseFile, line)) {
        // parse line
        // format:
        //   position=[x (float)],[y (float)]
        //   velocity=[vx (float)|nan],[vy (float)|nan]
        //   orientation=[orientation (float)|nan]°
        // Note: match floats against anything, since lexical_cast handles errors
        //   quite good (yielding NaN).
        //   float expression could be: [+-]?\d*\.?\d*
        //   (doesn't handle scientific notation)

        static const boost::regex patternPosition("position=(?<x>[^\n]*),(?<y>[^\n]*)");
        static const boost::regex patternVelocity("velocity=(?<x>[^\n]*),(?<y>[^\n]*)");
        static const boost::regex patternOrientation("orientation=(?<orientation>[^\n]*)°");

        // try to parse each pattern
        try {
            boost::smatch reMatches;
            if(boost::regex_match(line, reMatches, patternPosition)) {
                sampleInfo.pose.x = boost::lexical_cast<double>(reMatches[1]);
                sampleInfo.pose.y = boost::lexical_cast<double>(reMatches[2]);
            }
            else if(boost::regex_match(line, reMatches, patternVelocity)) {
                // compare to NaN (ignoring case)
                std::string velocityXString = reMatches[1];
                boost::algorithm::to_lower(velocityXString);
                std::string velocityYString = reMatches[2];
                boost::algorithm::to_lower(velocityYString);

                sampleInfo.velocity.x = (velocityXString == "nan" || velocityXString == "-nan") ? NaN : boost::lexical_cast<double>(velocityXString);
                sampleInfo.velocity.y = (velocityYString == "nan" || velocityYString == "-nan") ? NaN : boost::lexical_cast<double>(velocityYString);
            }
            else if(boost::regex_match(line, reMatches, patternOrientation)) {
                // compare to NaN (ignoring case)
                std::string orientationString = reMatches[1];
                boost::algorithm::to_lower(orientationString);

                sampleInfo.thetaDeg = (orientationString == "nan" || orientationString == "-nan") ? NaN : boost::lexical_cast<double>(reMatches[1]);
            }
            else {
                ROS_WARN_STREAM("Unmatchable line in pose file '" << poseFilename << "': " << line);
                return false;
            }
        }
        catch(const boost::bad_lexical_cast& exception) {
            ROS_WARN_STREAM("Invalid entries in pose file '" << poseFilename << "' (" << exception.what() << "): " << line);
            return false;
        }
    }

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
// Source: http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat createHOGDescriptorVisualImage(cv::Mat& origImg, std::vector<float>& descriptorValues, cv::Size winSize, cv::Size cellSize, int scaleFactor, double vizFactor)
{
    cv::Mat visualImage;
    cv::resize(origImg, visualImage, cv::Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
    cv::cvtColor(visualImage, visualImage, CV_GRAY2RGB);
    const int gradientBinSize = 9;

    // dividing 180° into 9 bins, how large (in rad) is one bin?
    const float radRangeForOneBin = M_PI / gradientBinSize; 
 
    // prepare data structure: 9 orientation / gradient strenghts for each cell
    const int cells_in_x_dir = winSize.width / cellSize.width;
    const int cells_in_y_dir = winSize.height / cellSize.height;
    int gradientStrengthsSizes[] = {cells_in_y_dir, cells_in_x_dir, gradientBinSize};
    cv::Mat gradientStrengths(3, gradientStrengthsSizes, CV_32SC1, cv::Scalar(0));
    cv::Mat cellUpdateCounter(cells_in_y_dir, cells_in_x_dir, CV_32FC1, cv::Scalar(0));
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++) {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++) {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++) {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3) {
                    cellx++;
                    celly++;
                }
 
                for (int bin=0; bin<gradientBinSize; bin++) {
                    float gradientStrength = descriptorValues[descriptorDataIdx];
                    descriptorDataIdx++;
 
                    gradientStrengths.at<float>(celly, cellx, bin) += gradientStrength;
 
                } // for (all bins) 
 
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter.at<int>(celly, cellx)++;

            } // for (all cells)
        } // for (all block x pos)
    } // for (all block y pos)
 
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++) {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++) {
            float nrUpdatesForThisCell = static_cast<float>(cellUpdateCounter.at<int>(celly, cellx));
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths.at<float>(celly, cellx, bin) /= nrUpdatesForThisCell;
        }
    }

    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++) {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++) {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;
 
            int mx = drawX + cellSize.width/2;
            int my = drawY + cellSize.height/2;
 
            cv::rectangle(visualImage, cv::Point(drawX*scaleFactor,drawY*scaleFactor), cv::Point((drawX+cellSize.width)*scaleFactor, (drawY+cellSize.height)*scaleFactor), CV_RGB(100,100,100), 1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++) {
                float currentGradStrength = gradientStrengths.at<float>(celly, cellx, bin);
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float maxVecLen = cellSize.width/2;
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * vizFactor;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * vizFactor;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * vizFactor;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * vizFactor;
 
                // draw gradient visual_imagealization
                cv::line(visualImage, cv::Point(x1*scaleFactor,y1*scaleFactor), cv::Point(x2*scaleFactor,y2*scaleFactor), CV_RGB(0, 255, 0), 1);
            } // for (all bins)
        } // for (cellx)
    } // for (celly)
 
    return visualImage;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool calculateHOGDescriptor(const std::string& filePath, std::vector<float>& featureVector, const std::string& maskSuffix, const std::string& visualizationName, bool useMasks) {
    cv::Mat imageData = cv::imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
    if(!imageData.data) {
        ROS_WARN_STREAM("Failed to read image, skipping sample: " << filePath);
        return false;
    }

    // FIXME: Make configurable
    cv::Size winSize(g_winSizeX, g_winSizeY);
    cv::Size trainingPadding(0, 0);
    cv::Size winStride(8, 8);

    // Use mask files to remove non-person areas?
    if(g_applyMasks && useMasks) {
        std::string maskFilename = filePath;
        boost::replace_all(maskFilename, "_depthcrop", maskSuffix); // FIXME: dirty hack
        boost::replace_all(maskFilename, "_crop", maskSuffix);
        cv::Mat maskData = cv::imread(maskFilename, CV_LOAD_IMAGE_GRAYSCALE) / 255;
        if(!maskData.data) {
            ROS_WARN_STREAM("Failed to read mask: " << maskFilename);
            return false;
        }

        maskData.convertTo(maskData, CV_32F);
        cv::GaussianBlur(maskData, maskData, cv::Size(21, 21), 11.0);

        cv::Mat imageDataFloat;
        imageData.convertTo(imageDataFloat, CV_32F);
        imageDataFloat /= 255;
        cv::Mat selectedImageData = 255 * imageDataFloat.mul(maskData);
        selectedImageData.convertTo(imageData, imageData.type());
    }

    // resize image to window size
    resize(imageData, imageData, winSize);

    // compute HOG features
    cv::HOGDescriptor hog;
    hog.winSize = winSize;
    std::vector<cv::Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);

    // visualization
    if(g_visualization) {
        cv::Mat hogImage = createHOGDescriptorVisualImage(imageData, featureVector, hog.winSize, winStride, 5, 5);
        cv::imshow(visualizationName.c_str(), hogImage);
    }

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void calculateFeaturesOnDataset(int fold, bool useValidationSet, cv::Mat& labels, cv::Mat& features, cv::Mat& sampleIdx, cv::Mat& missingDataMask,
    std::vector<std::string>& sampleFilenames, std::vector<SampleInfo>& sampleInfos)
{
    //
    // Load training data set and calculate features
    //
    ROS_INFO_STREAM("Initializing " << (useValidationSet ? "validation" : "training") << " set for fold " << fold << " of category '" << g_category << "'...");

    std::stringstream trainFilename;
    trainFilename << ros::package::getPath("top_down_classifier") << "/data/" << g_category << "/fold" << std::setw(3) << std::setfill('0') << fold << "/" << (useValidationSet ? "val" : "train") << ".txt";
    std::ifstream listFile(trainFilename.str().c_str());

    string sampleFilename;
    int label;
    size_t numSamples = 0;
    set<int> labelSet;

    // Skip comments at beginning of file
    std::string commentString;
    const size_t numCommentLinesAtBeginning = 2;
    for(size_t i = 0; i < numCommentLinesAtBeginning; i++) std::getline(listFile, commentString);

    // See how many sample files we have
    while (listFile >> sampleFilename >> label)
    {
        numSamples++;
    }

    // Go back to start of list file
    listFile.clear();
    listFile.seekg(0, std::ios::beg);
    for(size_t i = 0; i < numCommentLinesAtBeginning; i++) std::getline(listFile, commentString);

    // Initialize OpenCV matrices
    labels.create(numSamples, 1, CV_32SC1);
    labels.setTo(cv::Scalar(99999));
  
    sampleIdx.create(numSamples, 1, CV_8UC1);
    sampleIdx.setTo(cv::Scalar(0));

    bool featureMatrixInitialized = false;

    // Clear other fields
    g_numNegativeSamples = g_numPositiveSamples = 0;
    sampleFilenames.clear();
    sampleInfos.clear();


    ROS_INFO_STREAM("Loading " << numSamples << " samples to calculate features...");

    // Now start calculating features
    size_t numSkippedLowQualitySamples = 0;
    ros::WallRate rate(10);
    int goodSampleCounter = 0, totalSampleCounter = 0;
    while (listFile >> sampleFilename >> label && ros::ok())
    {
        // Show progress
        if(totalSampleCounter % (numSamples / 10) == 0) {
            ROS_INFO("%d %% of feature computations done...", int(totalSampleCounter / (float)numSamples * 100.0f + 0.5f));
        }
        totalSampleCounter++;

        // Adjust paths
        // FIXME: Currently hardcoded
        boost::replace_all(sampleFilename, "/home/linder/Datasets/srl_dataset_clouds", "/media/srl_dataset_remote/srl_persons");

        // Load pose file
        std::string poseFilename = sampleFilename;
        boost::replace_all(poseFilename, "_cloud.pcd", "_pose.txt");

        SampleInfo sampleInfo;
        if(!loadSampleInfo(poseFilename, sampleInfo)) {
            ROS_WARN("Couldn't read pose file %s", poseFilename.c_str());
            continue;
        }       

        // Skip low-quality samples during training stage
        if(!useValidationSet) {
            /*
            if(hypot(sampleInfo.pose.x, sampleInfo.pose.y) < 1.0 || abs(sampleInfo.phi) > 30.0) {
                ROS_WARN("Skipping sample %s which does not meet goodness criteria.\nDist: %.1f  Phi: %.1f  #Points: %d", sampleFilename.c_str(), sampleInfo.sensorDistance, abs(sampleInfo.phi), (int)sampleInfo.numPoints);
                numSkippedLowQualitySamples++;
                continue;
            } 
            */     
        }

        // Compute HOG descriptors on color and depth image
        bool combineColorAndDepth = g_modality == "combined";
        bool useColor = g_modality == "color" || combineColorAndDepth;
        bool useDepth = g_modality == "depth" || combineColorAndDepth;
        bool useMasks = !useValidationSet; // only apply masks in training phase, if enabled

        std::vector<float> colorFeatureVector, depthFeatureVector, finalFeatureVector;
        if(useColor) {
            std::string rgbFilename = sampleFilename;
            boost::replace_all(rgbFilename, "_cloud.pcd", "_crop.png");
            if(!calculateHOGDescriptor(rgbFilename, colorFeatureVector, "_maskcrop_color", "Color", useMasks)) continue;
        }
        if(useDepth) {
            std::string depthFilename = sampleFilename;
            boost::replace_all(depthFilename, "_cloud.pcd", "_depthcrop.png");
            if(!calculateHOGDescriptor(depthFilename, depthFeatureVector, "_maskcrop_depth", "Depth", useMasks)) continue;
        }

        // Concatenate color and depth feature vectors if required
        if(combineColorAndDepth) {
            finalFeatureVector = colorFeatureVector;
            finalFeatureVector.insert(finalFeatureVector.end(), depthFeatureVector.begin(), depthFeatureVector.end());
        }
        else {
            if(useColor) finalFeatureVector = colorFeatureVector;
            if(useDepth) finalFeatureVector = depthFeatureVector;
        }

        // Initialize feature and missing data matrix, only happens at first successfully processed sample
        if(!featureMatrixInitialized) {
            features.create(numSamples, finalFeatureVector.size(), CV_32FC1);
            features.setTo(cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
          
            missingDataMask.create(numSamples, finalFeatureVector.size(), CV_8UC1);
            missingDataMask.setTo(cv::Scalar(1));

            featureMatrixInitialized = true;
        }

        // Write into OpenCV matrix
        for(size_t f = 0; f < finalFeatureVector.size(); f++) {
            features.at<float>(goodSampleCounter, f) = finalFeatureVector[f];
            missingDataMask.at<unsigned char>(goodSampleCounter, f) = !std::isfinite(finalFeatureVector[f]) ? 1 : 0;
        }

        // Everything good! Now store meta data and name of input sample (for dump of results later on)
        sampleInfos.push_back(sampleInfo);
        sampleFilenames.push_back(sampleFilename);

        // Mark sample as active
        sampleIdx.at<unsigned char>(goodSampleCounter) = 1;

        // Store label
        labels.at<int>(goodSampleCounter) = label;
        labelSet.insert(label);
        
        if(label > 0) g_numPositiveSamples++;
        else g_numNegativeSamples++;
        
        // Prepare for next sample
        goodSampleCounter++;
        ros::spinOnce();

        // Update visualization, if enabled
        if(g_visualization) {
            cv::waitKey(1);
            rate.sleep();
        }
    }

    size_t numFailedSamples = totalSampleCounter-goodSampleCounter;
    ROS_INFO("Number of samples that failed to load: %zu (%.1f%%)", numFailedSamples,  numFailedSamples / float(totalSampleCounter) * 100.0f);
    ROS_ASSERT_MSG(goodSampleCounter > 0, "Did not find any valid training/test samples");

    // Optional: Dump feature values + labels into CSV file for off-line analysis
    stringstream exportPrefix;
    exportPrefix << g_category << "_fold" << fold << "_" << (useValidationSet ? "val" : "train") << "_";

    if(g_dumpFeatures) {
        ROS_INFO("Dumping features into file for optional off-line analysis...");
        std::ofstream featureFile( (exportPrefix.str() + "features.txt").c_str() );
        std::ofstream missingFile( (exportPrefix.str() + "missing.txt").c_str() );
        std::ofstream labelFile( (exportPrefix.str() + "labels.txt").c_str() );

        featureFile << features;
        missingFile << missingDataMask;
        labelFile << labels;
    }

    // Optional: Export features into numpy (.npy) file
    if(!g_numpyExportFolder.empty()) {
        /*
        ROS_INFO("Dumping features into numpy format for off-line analysis...");

        const unsigned int featuresShape[] = { features.rows, features.cols };
        cnpy::npy_save( g_numpyExportFolder + "/" + exportPrefix.str() + "features.npy", (float*)features.data, featuresShape, 2, "w" );

        const unsigned int missingShape[] = { missingDataMask.rows, missingDataMask.cols };
        cnpy::npy_save( g_numpyExportFolder + "/" + exportPrefix.str() + "missing.npy", (unsigned char*)missingDataMask.data, missingShape, 2, "w" );

        const unsigned int labelsShape[] = { labels.rows, labels.cols };
        cnpy::npy_save( g_numpyExportFolder + "/" + exportPrefix.str() + "labels.npy", (int*)labels.data, labelsShape, 2, "w" );
        */
    }


    std::stringstream ss;
    foreach(int label, labelSet) {
        ss << label << " ";
    }
    ROS_INFO_STREAM("Labels in input data: " << ss.str());
    ROS_INFO_STREAM("Feature computation complete on " << (useValidationSet ? "validation" : "training") << " set!");

    if(!useValidationSet) {
        ROS_INFO_STREAM("Number of samples skipped in training phase due to low quality: " << numSkippedLowQualitySamples << " ("
                        << std::fixed << std::setprecision(2) << (numSkippedLowQualitySamples / (float)numSamples * 100.0f) << "%)");
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "hog_attribute_training");
    ros::NodeHandle nodeHandle("");
    ros::NodeHandle privateHandle("~");

    privateHandle.param<bool>("dump_features", g_dumpFeatures, false);

    int initialFold, numFolds;
    privateHandle.param<int>("fold", initialFold, 0);
    privateHandle.param<int>("num_folds", numFolds, 1); // set to e.g. 5 if there are five splitX folders in "data" (cross-validation)
    
    privateHandle.param<string>("npy_export_folder", g_numpyExportFolder, "");
    privateHandle.param<string>("category", g_category, "");
    ROS_ASSERT_MSG(!g_category.empty(), "_category must be specified (e.g. 'gender')");

    privateHandle.param<std::string>("modality", g_modality, "combined"); // "combined", "color" or "depth"
    privateHandle.param<bool>("visualization", g_visualization, false); // visualize resulting HOG descriptors?
    privateHandle.param<bool>("apply_masks", g_applyMasks, false); // remove background using foreground/background segmentation masks?
    privateHandle.param<int>("win_size_x", g_winSizeX, 64); // HOG descriptor window size in X direction
    privateHandle.param<int>("win_size_y", g_winSizeY, 128); // HOG descriptor window size in X direction
     
    ROS_ASSERT(g_modality == "color" || g_modality == "depth" || g_modality == "combined");
    ROS_INFO_STREAM("Apply foreground/background segmentation masks: " << (g_applyMasks ? "YES" : "NO"));
    ROS_INFO_STREAM("Using a HOG descriptor window size of " << g_winSizeX << "x" << g_winSizeY << " px");
    ROS_INFO_STREAM("Active modalities: " << g_modality);
    
    //omp_set_num_threads(1);


    //
    // Perform training across all selected training/test set folds
    //
    std::vector<double> testAccuracyPerFold;
    std::vector< cv::Ptr<CvSVM> > learnedClassifiers;

    double bestTestAccuracySoFar = 0;
    cv::Ptr<CvSVM> bestClassifierSoFar;

    std::stringstream ss;

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
        std::vector<std::string> sampleFilenames;
        std::vector<SampleInfo> sampleInfos;
        calculateFeaturesOnDataset(fold, false, labels, features, sampleIdx, missingDataMask, sampleFilenames, sampleInfos);
        


        //
        // Train classifier
        //

        ROS_INFO_STREAM("");
        ROS_INFO_STREAM("=== STARTING TRAINING PHASE: LEARN CLASSIFIER! ===");

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

        
        CvSVMParams params;
        params.svm_type    = CvSVM::C_SVC;
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

        ROS_INFO_STREAM("Training classifier... this may take a while!");
        
        cv::Ptr<CvSVM> classifier(new CvSVM);
        if(!classifier->train_auto(features, labels, cv::Mat(), sampleIdx, params)) {
            ROS_ERROR("Training of classifier failed for unknown reason!");
        }

        learnedClassifiers.push_back(classifier);


        //
        // Test classifier
        //

        ROS_INFO_STREAM("");
        ROS_INFO_STREAM("=== STARTING TESTING PHASE: TESTING CLASSIFIER ON TRAINING SET! ===");
        ss.str(""); ss << "hog_train_results_fold" << fold << ".csv";
        double trainAccuracy = testClassifier(*classifier, labels, features, sampleIdx, missingDataMask, sampleFilenames, sampleInfos, ss.str());
        ROS_INFO("Accuracy on training set: %.2f%%", trainAccuracy * 100.0);

        // Load test set
        ROS_INFO_STREAM("");
        ROS_INFO_STREAM("=== STARTING TESTING PHASE: TESTING CLASSIFIER ON TEST SET! ===");
        
        calculateFeaturesOnDataset(fold, true, labels, features, sampleIdx, missingDataMask, sampleFilenames, sampleInfos);  // true = use validation set

        ss.str(""); ss << "hog_test_results_fold" << fold << ".csv";
        double testAccuracy = testClassifier(*classifier, labels, features, sampleIdx, missingDataMask, sampleFilenames, sampleInfos, ss.str());
        ROS_INFO("Accuracy on test set: %.2f%%", testAccuracy * 100.0);

        if(testAccuracy > bestTestAccuracySoFar) {
            bestTestAccuracySoFar = testAccuracy;
            bestClassifierSoFar = classifier;
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
    // Save best learned classifier
    //

    ROS_INFO_STREAM("");
    ROS_INFO_STREAM("=== SAVING CLASSIFIER WHICH PERFORMED BEST IN TESTING! ==="); 

    cv::FileStorage fileStorage("hog_classifier.yaml", cv::FileStorage::WRITE);

    // Names of all features
    fileStorage << "category" << g_category;

    // Write the classifier itself
    bestClassifierSoFar->write(*fileStorage, "classifier");

    // Close file
    fileStorage.release();


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
