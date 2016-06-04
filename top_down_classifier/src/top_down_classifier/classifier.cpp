#include "classifier.h"
#include "cloud_preprocessing.h"

#include <ros/ros.h>
#include <algorithm>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;

// Helpers for sorting volumes by descending quality (for visualization of best N volumes)
struct VolumeQuality {
    size_t volumeIndex;
    float quality;
};
struct VolumeQualityComparator {
    bool operator() (const VolumeQuality& lhs, const VolumeQuality& rhs) const {
       return lhs.quality > rhs.quality;
    }
};



TopDownClassifier::TopDownClassifier()
    : m_expectedFeatureVectorLength(0), m_numActiveFeatureTypes(0), m_numTotalFeatureTypes(0), m_minPoints(4), m_isOptimized(false)
{}


pcl::PointXYZ readCoords(cv::FileNode coords)
{
    pcl::PointXYZ result;
    coords["x"] >> result.x;
    coords["y"] >> result.y;
    coords["z"] >> result.z;
    //ROS_INFO_STREAM("Read coords: " << result);
    return result;
}

Volume readVolume(cv::FileNode volumeNode)
{
    return Volume( readCoords(volumeNode["minCoords"]), readCoords(volumeNode["maxCoords"]) ); 
}


void TopDownClassifier::init(const string& modelFilename)
{
    ROS_INFO_STREAM("Opening " << modelFilename << " to read learned model...");
    cv::FileStorage fileStorage(modelFilename, cv::FileStorage::READ);
    CvFileStorage* fileStorageInternal = *fileStorage; // for easier debugging, see fileStorageInternal.lineno value
    ROS_ASSERT_MSG(fileStorage.isOpened(), "Failed to open learned model %s", modelFilename.c_str());

    // Read classifier category (attribute which it was trained on)
    fileStorage["category"] >> m_category;
    if(m_category.empty()) m_category = "gender";  // for old model files
    ROS_INFO_STREAM("Model was trained on attribute " << m_category);

    // Check if this is an optimized classifier or not; if not, output warning
    fileStorage["optimized"] >> m_isOptimized;
    if(!m_isOptimized) {
        ROS_WARN("*** The provided learned classifier model is not optimized, which may result in significantly reduced runtime performance. ***\n"
                 "Run optimize_classifier_model.py FILENAME.yaml to eliminate unneeded feature vector dimensions!");
    }


    // Get parent volume
    cv::FileNode parentVolumeNode = fileStorage["parent_volume"];
    if(parentVolumeNode.isNone()) {
        ROS_WARN("Learned classifier model doesn't specify parent volume that it was trained on. Assuming defaults.");
        pcl::PointXYZ minCoords(-0.3, -0.3, 0), maxCoords(0.3, 0.3, 1.8);
        m_parentVolume = Volume(minCoords, maxCoords);
    }
    else m_parentVolume = readVolume(parentVolumeNode);

    // NOTE: Parent volume size is used to calculate some feature values (e.g. point ratio). It
    // is therefore a training parameter and must not be modified after training has been completed.
    pcl::PointXYZ parentVolumeSize = m_parentVolume.getSize();
    ROS_INFO("Parent volume is of size %.1f x %.1f x %.1f m!", parentVolumeSize.x, parentVolumeSize.y, parentVolumeSize.z);

    // Get active features
    ROS_INFO("Reading active features...");
    vector<string> availableFeatures = m_featureCalculator.getFeatureNames();
    m_numTotalFeatureTypes = availableFeatures.size();

    m_numActiveFeatureTypes = 0;
    vector<string> activeFeatures;

    cv::FileNode featureNames = fileStorage["feature_names"];
    for(cv::FileNodeIterator it = featureNames.begin(); it != featureNames.end(); ++it) {
        string featureName; (*it) >> featureName;
        activeFeatures.push_back(featureName);

        bool featureFound = false;
        for(size_t f = 0; f < availableFeatures.size() && !featureFound; f++) {
            if(availableFeatures[f] == featureName) {
                m_numActiveFeatureTypes++;
                featureFound = true;
            }
        }

        ROS_ASSERT_MSG( find(availableFeatures.begin(), availableFeatures.end(), featureName) != availableFeatures.end(),
            "A feature named '%s' does not exist!", featureName.c_str());
    }

    // Get active volumes
    ROS_INFO("Reading active volumes...");
    m_volumes = vector<Volume>();

    cv::FileNode tessellationVolumes = fileStorage["tessellation_volumes"];
    for(cv::FileNodeIterator it = tessellationVolumes.begin(); it != tessellationVolumes.end(); ++it) {
        Volume volume = readVolume(*it);
        m_volumes.push_back(volume);
    }

    // Get active features per volume (m_activeFeaturesPerVolume)
    ROS_INFO("Reading active features per volume...");
    m_activeFeaturesPerVolume = vector< vector<bool> >();
    
    cv::FileNode activeFeaturesPerVolume = fileStorage["active_features_per_volume"];
    if(!activeFeaturesPerVolume.isNone()) {
        for(cv::FileNodeIterator it = activeFeaturesPerVolume.begin(); it != activeFeaturesPerVolume.end(); ++it)
        {
            // YAML file lists only active features per volume; assume all others to be inactive 
            vector<bool> featureIsActive(m_numTotalFeatureTypes, false);
            cv::FileNode activeFeaturesInCurrentVolume = *it;
            for(cv::FileNodeIterator it = activeFeaturesInCurrentVolume.begin(); it != activeFeaturesInCurrentVolume.end(); ++it) {
                int featureIndex = (int) *it;
                ROS_ASSERT(featureIndex < m_numActiveFeatureTypes);
                featureIsActive[featureIndex] = true;
            }
            m_activeFeaturesPerVolume.push_back(featureIsActive);
        }
    }
    else {
        // Assume all features are active
        ROS_INFO("All features per volume are active!");
        for(size_t v = 0; v < m_volumes.size(); v++) {
            m_activeFeaturesPerVolume.push_back(vector<bool>(m_numTotalFeatureTypes, true));
        }
    }

    // Count number of active features per volume, and total variable count
    size_t numActiveVars = 0;
    for(size_t v = 0; v < m_activeFeaturesPerVolume.size(); v++) {
        m_numAccumulatedVarsBeforeVolume.push_back(numActiveVars);
        m_numActiveFeaturesPerVolume.push_back(0);

        vector<bool> bitmask = m_activeFeaturesPerVolume[v];
        for(size_t b = 0; b < bitmask.size(); b++) {
            if(bitmask[b]) {
                m_numActiveFeaturesPerVolume[v]++;
                numActiveVars++;
            }
        }
    }

    // Get feature vector entries, to be able to sort volumes or features by quality
    std::vector<VolumeQuality> volumeQualities;
    
    ROS_INFO("Reading feature vector entries...");
    cv::FileNode featureVectorEntries = fileStorage["feature_vector_entries"];
    for(cv::FileNodeIterator it = featureVectorEntries.begin(); it != featureVectorEntries.end(); ++it) {
        cv::FileNode featureVectorEntry = *it;
        int overallVolumeIndex = (int) featureVectorEntry["overallVolumeIndex"];
        int featureIndex = (int) featureVectorEntry["featureIndex"];

        float splitQuality = std::numeric_limits<float>::min();
        if(!featureVectorEntry["quality"].isNone()) splitQuality = (float) featureVectorEntry["quality"];

        VolumeQuality volumeQuality;
        volumeQuality.quality = splitQuality;
        volumeQuality.volumeIndex = overallVolumeIndex;
        volumeQualities.push_back(volumeQuality);
    }

    // Sort volume qualities by quality, then extract indices of best volumes
    std::stable_sort(volumeQualities.begin(), volumeQualities.end(), VolumeQualityComparator());
    foreach(const VolumeQuality& volumeQuality, volumeQualities) {
        m_bestVolumeIndices.push_back(volumeQuality.volumeIndex);
    }

    // Get number of minimum required points per volume
    m_minPoints = (int) fileStorage["min_points"];

    // Get expected feature vector length
    m_expectedFeatureVectorLength = (int) fileStorage["feature_vector_length"];
    ROS_ASSERT(numActiveVars == m_expectedFeatureVectorLength);

    // Get pre-processing parameters
    ROS_INFO("Reading cloud pre-processing parameters...");
    
    m_scaleZto = 0.0f;
    if(!fileStorage["scale_z_to"].isNone()) m_scaleZto = (float) fileStorage["scale_z_to"];

    m_cropZmin = -std::numeric_limits<float>::infinity();
    if(!fileStorage["crop_z_min"].isNone()) m_cropZmin = (float) fileStorage["crop_z_min"];

    m_cropZmax = +std::numeric_limits<float>::infinity();
    if(!fileStorage["crop_z_max"].isNone()) m_cropZmax = (float) fileStorage["crop_z_max"];

    ROS_INFO("Input cloud scaling factor in z direction is %.3f", m_scaleZto > 0 ? m_scaleZto : 1.0f);
    ROS_INFO("Cropping input clouds in z direction between %.3f and %.3f", m_cropZmin, m_cropZmax);

    // Get Adaboost classifier
    ROS_INFO("Reading classifier...");
    cv::FileNode classifier = fileStorage["classifier"];
    m_adaboost.read(*fileStorage, *classifier);

    ROS_INFO_STREAM("Model contains " << m_volumes.size() << " tessellation volume(s) and has a feature vector of length " << m_expectedFeatureVectorLength);
    ROS_INFO("Learned model has been loaded successfully!");
}


class_label TopDownClassifier::classify(PointCloud::Ptr personCloud, double* weightedSum) const
{
    ROS_ASSERT_MSG(m_expectedFeatureVectorLength > 0, "Classifier has not been initialized (no weights file loaded)!");

    // Pre-process cloud
    // IMPORTANT: This modifies the input cloud! Otherwise we'd have to copy the cloud, which would be slow
    scaleAndCropCloudToTargetSize(personCloud, m_scaleZto, m_cropZmin, m_cropZmax);

    // Compute features for this cloud
    cv::Mat featureVector, missingDataMask;
    calculateFeatures(*personCloud, featureVector, missingDataMask);

    // Run classifier
    bool returnSum = weightedSum != NULL;
    float result = m_adaboost.predict(featureVector, missingDataMask, cv::Range::all(), false, returnSum);

    if(returnSum) {
        *weightedSum = result; // result contains sum of votes by weak classifiers

        // From opencv/modules/ml/src/boost.cpp, predict()
        // Maps weighted sum to a class label, since we need both
        int cls_idx = result >= 0;
        const CvDTreeTrainData* data = m_adaboost.get_data();
        const int* vtype = data->var_type->data.i;
        const int* cmap = data->cat_map->data.i;
        const int* cofs = data->cat_ofs->data.i;
        return (float)cmap[cofs[vtype[data->var_count]] + cls_idx];  
    }
    else return (int) result; // result contains class label
}


void TopDownClassifier::calculateFeatures(const PointCloud& personCloud, cv::Mat& featureVector, cv::Mat& missingDataMask) const
{
    // Allocate memory for resulting feature vector
    featureVector.create(1, m_expectedFeatureVectorLength, CV_32FC1);
    featureVector.setTo(cv::Scalar(numeric_limits<double>::quiet_NaN()));

    missingDataMask.create(1, m_expectedFeatureVectorLength, CV_8UC1);
    missingDataMask.setTo(cv::Scalar(1));

    // Calculate features for each volume
    #pragma omp parallel for schedule(dynamic)
    for(size_t v = 0; v < m_volumes.size(); v++)
    {
        // Get correct volume
        const Volume& volume = m_volumes[v];

        // Get indices of all points inside current volume
        std::vector<int> indicesInsideVolume;
        volume.getPointsInsideVolume(personCloud, PointCloud::Ptr(), &indicesInsideVolume);

        // Only if there are sufficient points inside volume
        const size_t expectedVolumeFeatureVectorLength = m_numActiveFeaturesPerVolume[v];
        vector<double> volumeFeatureVector;
        if(indicesInsideVolume.size() >= m_minPoints) {
            ROS_ASSERT_MSG(m_activeFeaturesPerVolume[v].size() == m_featureCalculator.getFeatureCount(), "Active features mask for volume should have %zu element(s) but has %zu!",
                m_featureCalculator.getFeatureCount(), m_activeFeaturesPerVolume[v].size());
            
            m_featureCalculator.calculateFeatures(m_parentVolume, personCloud, indicesInsideVolume,
                m_activeFeaturesPerVolume[v], volumeFeatureVector); 
        }
        else volumeFeatureVector = vector<double>(expectedVolumeFeatureVectorLength, numeric_limits<double>::quiet_NaN());

        // Copy feature values into right spot of sample's overall feature vector
        ROS_ASSERT_MSG(volumeFeatureVector.size() == expectedVolumeFeatureVectorLength, "Expected volume feature vector for volume %zu to have %zu element(s), but got %zu!",
            v, expectedVolumeFeatureVectorLength, volumeFeatureVector.size());
            
        for(size_t f = 0; f < volumeFeatureVector.size(); f++) { // for each feature...
            size_t varIndex = m_numAccumulatedVarsBeforeVolume[v] + f;

            // NOTE: Avoiding this copy and directly referencing featureVector via pointer doesn't improve performance
            // FIXME: This may lead to the "false sharing" problem with OMP, negatively impacting performance.
            // Instead, we could copy entire slices of the full feature vector instead of assigning single values
            // on an individual basis. 
            featureVector.at<float>(varIndex) = volumeFeatureVector[f];
            missingDataMask.at<unsigned char>(varIndex) = !isfinite(volumeFeatureVector[f]) ? 1 : 0;
        }
    }
}
