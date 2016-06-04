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
    ///
    /// IMPORTANT: Due to the pre-processing (scaling, cropping) performed by the learned classifier, the input personCloud might be modified by this method!!!
    /// If this is not intended, please create a copy of the cloud before passing it to this function.
    class_label classify(PointCloud::Ptr personCloud, double* weightedSum = NULL) const;

    /// Returns all tessellation volumes that are used by this classifier. The volumes must not be modified by the caller.
    const std::vector<Volume>& getVolumes() const { return m_volumes; }

    /// Return parent volume loaded from classifier file
    const Volume& getParentVolume() const { return m_parentVolume; }

    /// Returns whether the learned classifier model has been optimized such that it contains
    /// only the relevant feature dimensions instead of the whole, high-dimensional feature vector with all possible tessellations.
    bool isOptimized() const { return m_isOptimized; }

    /// Returns the name of the category (or attribute) that this classifier was trained on, e.g. "gender".
    const std::string& getCategory() const { return m_category; }

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
    float m_scaleZto, m_cropZmin, m_cropZmax;
    std::string m_category;
};


#endif // TOP_DOWN_CLASSIFIER_H_