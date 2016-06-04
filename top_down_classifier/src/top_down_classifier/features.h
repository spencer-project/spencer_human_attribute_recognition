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