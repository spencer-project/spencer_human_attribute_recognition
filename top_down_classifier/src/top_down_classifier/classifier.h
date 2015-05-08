#ifndef TOP_DOWN_CLASSIFIER_H_
#define TOP_DOWN_CLASSIFIER_H_

#include "volume.h"
#include "common.h"
#include "features.h"

#include <opencv2/opencv.hpp>


/// Forward declarations
class CvBoost;

/// A trained top-down classifier. Loads training weights from a .yaml file.
/// For training a classifier, see train_node.cpp.
class TopDownClassifier {
public:
    TopDownClassifier();

    /// Initializes the classifier. Must be called before classification can occur.
    /// modelFilename is path to a YAML file containing a trained classifier.
    void init(const std::string& modelFilename);

    /// Classifies a given point cloud. Assumes that the point-cloud contains only a single person and
    /// has been oriented such that the sensor is looking into the +x direction, +y is left and +z is up.
    /// The persons median x and y coordinates should be centered around the origin, and the feet should be placed at the xy plane (z=0).
    /// If the weightedSum pointer is not NULL, the weighted sum of the weak classifiers will be stored in there.
    class_label classify(const PointCloud& personCloud, double* weightedSum = NULL) const;

    /// Returns all tessellation volumes that are used by this classifier. The volumes must not be modified by the caller.
    const std::vector<Volume>& getVolumes() const { return m_volumes; }

    /// Return parent volume loaded from classifier file
    const Volume& getParentVolume() const { return m_parentVolume; }

    /// Returns whether the learned classifier model has been optimized such that it contains
    /// only the relevant feature dimensions instead of the whole, high-dimensional feature vector with all possible tessellations.
    bool isOptimized() const { return m_isOptimized; }

    /// Returns the indices of all tessellation volumes, but sorted after descending tree split (weak learner) quality.
    const std::vector<size_t> getIndicesOfBestVolumes() const {
        return m_bestVolumeIndices;
    }

private:
    void calculateFeatures(const PointCloud& personCloud, cv::Mat& featureVector, cv::Mat& missingDataMask) const;

    /// Instance variables
    size_t m_expectedFeatureVectorLength, m_numActiveFeatureTypes, m_numTotalFeatureTypes, m_minPoints;
    std::vector<Volume> m_volumes;
    std::vector< std::vector<bool> > m_activeFeaturesPerVolume;
    std::vector<size_t> m_numActiveFeaturesPerVolume, m_numAccumulatedVarsBeforeVolume;
    std::vector<size_t> m_bestVolumeIndices; // volume indices, sorted after highest quality
    CvBoost m_adaboost;
    FeatureCalculator m_featureCalculator;
    Volume m_parentVolume;
    bool m_isOptimized;
};


#endif // TOP_DOWN_CLASSIFIER_H_