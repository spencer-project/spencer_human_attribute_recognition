#ifndef FEATURES_H_
#define FEATURES_H_

#include "volume.h"

#include <vector>
#include <string>


class FeatureCalculator {
public:
    /// Constructor
    FeatureCalculator();

    /// Calculates features for all points belonging to the given indices in the point cloud.
    /// activeFeatures determines which of the features shall be calculated.
    /// The resulting vector will have a length equal to the number of "true" values in the activeFeatures vector.
    void calculateFeatures(const Volume& parentVolume, const PointCloud& fullCloud, const std::vector<int>& pointIndices,
        const std::vector<bool>& activeFeatures, std::vector<double>& featureVector) const;

    /// Returns a vector with names of the features, of the same size as featureVector.
    const std::vector<std::string>& getFeatureNames() { return m_featureNames; }

    /// Get the number of available features
    size_t getFeatureCount() const;

    /// Returns a bit mask to mark all available features as active
    const std::vector<bool>& maskAllFeaturesActive() {
        return m_allFeaturesMask;
    }

private:
    std::vector<bool> m_allFeaturesMask;
    std::vector<std::string> m_featureNames;
};



#endif // FEATURES_H_