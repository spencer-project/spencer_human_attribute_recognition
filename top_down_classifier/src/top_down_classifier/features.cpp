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

#include "features.h"

#include <Eigen/Eigenvalues> 
#include <ros/console.h>

#include <sstream>
#include <algorithm>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#define USE_STANDARD_FEATURES
#define USE_EXTENT_FEATURES
//#define USE_HSV_FEATURES
#define USE_YCBCR_FEATURES
//#define USE_RGB_FEATURES

enum Features {
    // NOTE: New features must be added to the end of the list, otherwise
    // existing learned classifier models will break!
#ifdef USE_STANDARD_FEATURES
    F_NUM_POINTS,
    F_SPHERICITY,
    F_FLATNESS,
    F_LINEARITY,
    F_STDDEV,
    F_KURTOSIS,
    F_MAD_MEDIAN,
    F_NORM_RES_PLANARITY,
    F_NUM_POINTS_RATIO,
#endif

    // Added after ICRA'15 paper: Width, depth, height of voxel content
#ifdef USE_EXTENT_FEATURES
    F_WIDTH,
    F_DEPTH,
    F_HEIGHT,
    F_ASPECT_RATIO_XY,
#endif

    // Added after ICRA'15 paper: HSV color features
#ifdef USE_HSV_FEATURES
    F_HSV_MEAN_H,
    F_HSV_MEAN_S,
    F_HSV_MEAN_V,
    F_HSV_STDDEV_H,
    F_HSV_STDDEV_S,
    F_HSV_STDDEV_V,
#endif

    // Added after ICRA'15 paper: YCbCr color features
#ifdef USE_YCBCR_FEATURES
    F_YCBCR_MEAN_Y,
    F_YCBCR_MEAN_CB,
    F_YCBCR_MEAN_CR,
    F_YCBCR_STDDEV_Y,
    F_YCBCR_STDDEV_CB,
    F_YCBCR_STDDEV_CR,
#endif

    // Added after ICRA'15 paper: RGB color features
#ifdef USE_RGB_FEATURES
    F_RGB_MEAN_R,
    F_RGB_MEAN_G,
    F_RGB_MEAN_B,
    F_RGB_STDDEV_R,
    F_RGB_STDDEV_G,
    F_RGB_STDDEV_B,
#endif

    F_COUNT, /* add new features before this element */
};


FeatureCalculator::FeatureCalculator() 
{
    std::vector<std::string> enabledFeatureSets;

    m_featureNames.resize(F_COUNT);
    m_allFeaturesMask = std::vector<bool>(F_COUNT, true);

#ifdef USE_STANDARD_FEATURES
    enabledFeatureSets.push_back("Standard geometric features");
    m_featureNames[F_NUM_POINTS]            = "Number of points";
    m_featureNames[F_SPHERICITY]            = "Sphericity";
    m_featureNames[F_FLATNESS]              = "Flatness";
    m_featureNames[F_LINEARITY]             = "Linearity";
    m_featureNames[F_STDDEV]                = "Standard deviation w.r.t. centroid";
    m_featureNames[F_KURTOSIS]              = "Kurtosis w.r.t. centroid";
    m_featureNames[F_MAD_MEDIAN]            = "Average deviation from median";
    m_featureNames[F_NORM_RES_PLANARITY]    = "Normalized residual planarity";
    m_featureNames[F_NUM_POINTS_RATIO]      = "Number of points ratio relative to total point count in cloud";  
#endif

#ifdef USE_EXTENT_FEATURES
    enabledFeatureSets.push_back("Geometric extent features");
    m_featureNames[F_WIDTH]                 = "Width of voxel content (extent in y direction)";  
    m_featureNames[F_DEPTH]                 = "Depth of voxel content (extent in x direction)";  
    m_featureNames[F_HEIGHT]                = "Height of voxel content (extent in z direction)";  
    m_featureNames[F_ASPECT_RATIO_XY]       = "Aspect ratio x/y";  
#endif

#ifdef USE_HSV_FEATURES
    enabledFeatureSets.push_back("HSV color features");
    m_featureNames[F_HSV_MEAN_H]            = "HSV color mean (hue)";
    m_featureNames[F_HSV_MEAN_S]            = "HSV color mean (saturation)";
    m_featureNames[F_HSV_MEAN_V]            = "HSV color mean (value)";
    m_featureNames[F_HSV_STDDEV_H]          = "HSV color standard deviation (hue)";
    m_featureNames[F_HSV_STDDEV_S]          = "HSV color standard deviation (saturation)";
    m_featureNames[F_HSV_STDDEV_V]          = "HSV color standard deviation (value)";
#endif       

#ifdef USE_YCBCR_FEATURES
    enabledFeatureSets.push_back("YCbCr color features");
    m_featureNames[F_YCBCR_MEAN_Y]          = "YCbCr color mean (Y)";
    m_featureNames[F_YCBCR_MEAN_CB]         = "YCbCr color mean (Cb)";
    m_featureNames[F_YCBCR_MEAN_CR]         = "YCbCr color mean (Cr)";
    m_featureNames[F_YCBCR_STDDEV_Y]        = "YCbCr color standard deviation (Y)";
    m_featureNames[F_YCBCR_STDDEV_CB]       = "YCbCr color standard deviation (Cb)";
    m_featureNames[F_YCBCR_STDDEV_CR]       = "YCbCr color standard deviation (Cr)";
#endif

#ifdef USE_RGB_FEATURES
    enabledFeatureSets.push_back("RGB color features");
    m_featureNames[F_RGB_MEAN_R]            = "RGB color mean (R)";
    m_featureNames[F_RGB_MEAN_G]            = "RGB color mean (G)";
    m_featureNames[F_RGB_MEAN_B]            = "RGB color mean (B)";
    m_featureNames[F_RGB_STDDEV_R]          = "RGB color standard deviation (R)";
    m_featureNames[F_RGB_STDDEV_G]          = "RGB color standard deviation (G)";
    m_featureNames[F_RGB_STDDEV_B]          = "RGB color standard deviation (B)";
#endif  

    std::stringstream ss;
    for(size_t i = 0; i < enabledFeatureSets.size(); i++) {
        ss << enabledFeatureSets[i];
        if(i < enabledFeatureSets.size() - 1) ss << ", ";
    }
    ROS_INFO_STREAM("Using the following sets of features: " << ss.str());
}

// NOTE: This code might be a bit hard to read since it contains optimizations to compute only what is absolutely
// required, given the bit mask of active features for this particular volume.
void FeatureCalculator::calculateFeatures(const Volume& parentVolume, const PointCloud& fullCloud, const std::vector<int>& pointIndices,
    const std::vector<bool>& activeFeatures, std::vector<double>& featureVector) const
{
    double pointCount = pointIndices.size();
        
    #ifdef USE_STANDARD_FEATURES
    {
        // Feature: Number of points
        if(activeFeatures[F_NUM_POINTS]) featureVector.push_back(pointCount);

        // Determine sample mean for scatter matrix
        bool needEigenvalues = activeFeatures[F_SPHERICITY] || activeFeatures[F_FLATNESS] || activeFeatures[F_LINEARITY] || activeFeatures[F_NORM_RES_PLANARITY];
        bool needMedian = activeFeatures[F_MAD_MEDIAN];
        bool needSampleMean = needEigenvalues || needMedian || activeFeatures[F_STDDEV] || activeFeatures[F_KURTOSIS];
        
        Eigen::Vector3d sampleMean = Eigen::Vector3d::Zero();
        std::vector<double> xs, ys, zs; // for feature: average deviation from median 
        
        if(needSampleMean && needMedian) {
            xs.reserve(pointIndices.size());
            ys.reserve(pointIndices.size());
            zs.reserve(pointIndices.size());

            foreach(int pointIndex, pointIndices) {
                const PointType& point = fullCloud.points[pointIndex];
                sampleMean += Eigen::Vector3d(point.x, point.y, point.z);

                // for feature: average deviation from median
                xs.push_back(point.x);
                ys.push_back(point.y);
                zs.push_back(point.z);
            }
            sampleMean /= pointCount;
        }
        else if(needSampleMean && !needMedian) {
            foreach(int pointIndex, pointIndices) {
                const PointType& point = fullCloud.points[pointIndex];
                sampleMean += Eigen::Vector3d(point.x, point.y, point.z);
            }
            sampleMean /= pointCount;
        }
        else if(!needSampleMean && needMedian) {
            xs.reserve(pointIndices.size());
            ys.reserve(pointIndices.size());
            zs.reserve(pointIndices.size());

            foreach(int pointIndex, pointIndices) {
                const PointType& point = fullCloud.points[pointIndex];
                
                // for feature: average deviation from median
                xs.push_back(point.x);
                ys.push_back(point.y);
                zs.push_back(point.z);
            }
        }

        Eigen::Vector3d componentWiseMedian;
        if(needMedian) {
            // for feature: average deviation from median
            // NOTE: Using std::nth_element() (which produces a partial sort) is significantly faster than std::sort()!
            std::nth_element(xs.begin(), xs.begin() + xs.size()/2, xs.end());
            std::nth_element(ys.begin(), ys.begin() + ys.size()/2, ys.end());
            std::nth_element(zs.begin(), zs.begin() + zs.size()/2, zs.end());
            componentWiseMedian = Eigen::Vector3d(xs[xs.size() / 2], ys[ys.size() / 2], zs[zs.size() / 2]);
        }

        // determine scatter matrix etc.
        Eigen::Matrix3d scatterMatrix = Eigen::Matrix3d::Zero();
        double sumOfSquaredDifferences = 0; // for feature: standard deviation
        double sumOfSquaredSquaredDifferences = 0; // for feature: kurtosis
        double sumOfDeviationsFromMedian = 0; // for feature: average deviation from median

        if(needEigenvalues || activeFeatures[F_STDDEV] || activeFeatures[F_KURTOSIS] || activeFeatures[F_MAD_MEDIAN])
        {
            foreach(int pointIndex, pointIndices) {
                const PointType& point = fullCloud.points[pointIndex];
                Eigen::Vector3d x(point.x, point.y, point.z);

                Eigen::Vector3d meanDiff = x - sampleMean;
                if(needEigenvalues) scatterMatrix += meanDiff * meanDiff.transpose();

                double meanDiffDotProduct = meanDiff.dot(meanDiff);
                if(activeFeatures[F_STDDEV])     sumOfSquaredDifferences += meanDiffDotProduct;
                if(activeFeatures[F_KURTOSIS])   sumOfSquaredSquaredDifferences += meanDiffDotProduct * meanDiffDotProduct;
                if(activeFeatures[F_MAD_MEDIAN]) sumOfDeviationsFromMedian += (x - componentWiseMedian).norm();
            }
        }

        Eigen::EigenSolver<Eigen::Matrix3d> eigensolver;
        Eigen::EigenSolver<Eigen::Matrix3d>::EigenvalueType eigenvalues;
        std::vector<double> sortedEigenvalues;

        if(needEigenvalues) {
            eigensolver = Eigen::EigenSolver<Eigen::Matrix3d>(scatterMatrix);
            eigenvalues = eigensolver.eigenvalues();
            sortedEigenvalues.push_back(eigenvalues(0).real());
            sortedEigenvalues.push_back(eigenvalues(1).real());
            sortedEigenvalues.push_back(eigenvalues(2).real());
            std::sort(sortedEigenvalues.begin(), sortedEigenvalues.end(), std::greater<double>());
            double sumOfEigenvalues = eigenvalues.real().sum();

            // Feature: Sphericity
            if(activeFeatures[F_SPHERICITY]) {
                double sphericity = 3 * sortedEigenvalues[2] / sumOfEigenvalues;
                featureVector.push_back(sphericity);
            }

            // Feature: Flatness
            if(activeFeatures[F_FLATNESS]) {
                double flatness = 2 * (sortedEigenvalues[1] - sortedEigenvalues[2]) / sumOfEigenvalues;
                featureVector.push_back(flatness);
            }

            // Feature: Linearity
            if(activeFeatures[F_LINEARITY]) {
                double linearity = (sortedEigenvalues[0] - sortedEigenvalues[1]) / sumOfEigenvalues;
                featureVector.push_back(linearity);
            }
        }

        // Feature: Standard deviation w.r.t. centroid
        double stddev;
        if(activeFeatures[F_STDDEV] || activeFeatures[F_KURTOSIS]) {
            stddev = sqrt(1/(pointCount - 1) * sumOfSquaredDifferences);
            if(activeFeatures[F_STDDEV]) featureVector.push_back(stddev);
        }

        // Feature: Kurtosis w.r.t. centroid
        if(activeFeatures[F_KURTOSIS]) {
            double kurtosis = sumOfSquaredSquaredDifferences / stddev;
            featureVector.push_back(kurtosis);
        }

        // Feature: Average deviation from median
        if(activeFeatures[F_MAD_MEDIAN]) {
            double averageDeviationFromMedian = sumOfDeviationsFromMedian / pointCount;
            featureVector.push_back(averageDeviationFromMedian);
        }

        // Feature: Normalized residual planarity
        // As described here: http://missingbytes.blogspot.de/2012/06/fitting-plane-to-point-cloud.html
        // in implicit form C+n*x = 0
        // the plane's normal vector n is the eigenvector associated with the smallest eigenvalue of the scatter matrix (=sortedEigenvalues(2))
        // and the center vector C is just the sample mean
        // FIXME: Might want to use SVD instead for numerical stability: http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points?rq=1
        if(activeFeatures[F_NORM_RES_PLANARITY]) {
            Eigen::EigenSolver<Eigen::Matrix3d>::EigenvectorsType eigenvectors = eigensolver.eigenvectors();
            Eigen::Vector3d normalVector = Eigen::Vector3d::Zero();
            for(int i = 0; i < eigenvectors.cols(); i++) {
                if(eigenvalues(i) == sortedEigenvalues[2]) {
                    normalVector = eigenvectors.col(i).real();
                    break;
                }
            }

            double distanceToOrigin = -normalVector.dot(sampleMean);

            double sumOfSquaredErrors = 0;
            foreach(int pointIndex, pointIndices) {
                const PointType& point = fullCloud.points[pointIndex];
                Eigen::Vector3d x(point.x, point.y, point.z);

                double residual = normalVector.dot(x) + distanceToOrigin;
                sumOfSquaredErrors += residual * residual;
            }

            double normalizedResidualPlanarity = sumOfSquaredErrors / pointCount;
            featureVector.push_back(normalizedResidualPlanarity);
        }

        // Feature: Number of points ratio
        //pcl::PointXYZ volumeSize = parentVolume.getSize();
        //double numberOfPointsRatio = pointCount / (parentVolume.x * parentVolume.y * parentVolume.z);
        if(activeFeatures[F_NUM_POINTS_RATIO]) {
            double numberOfPointsRatio = pointCount / (double)fullCloud.points.size(); // this assumes that full cloud contains only the person
            featureVector.push_back(numberOfPointsRatio);
        }
    }
    #endif

    //
    // Added after ICRA'15 paper: Width, depth, height of voxel content
    //
    #ifdef USE_EXTENT_FEATURES
    {
        bool needExtents = activeFeatures[F_WIDTH] || activeFeatures[F_DEPTH] || activeFeatures[F_HEIGHT] || activeFeatures[F_ASPECT_RATIO_XY];
        
        if(needExtents) {
            double xmin = +std::numeric_limits<double>::infinity(), xmax = -std::numeric_limits<double>::infinity();
            double ymin = +std::numeric_limits<double>::infinity(), ymax = -std::numeric_limits<double>::infinity();
            double zmin = +std::numeric_limits<double>::infinity(), zmax = -std::numeric_limits<double>::infinity();

            foreach(int pointIndex, pointIndices) {
                const PointType& point = fullCloud.points[pointIndex];
                
                if(point.x < xmin) xmin = point.x;
                if(point.y < ymin) ymin = point.y;
                if(point.z < zmin) zmin = point.z;

                if(point.x > xmax) xmax = point.x;
                if(point.y > ymax) ymax = point.y;
                if(point.z > zmax) zmax = point.z;
            }

            if(activeFeatures[F_WIDTH])  featureVector.push_back( ymax - ymin );
            if(activeFeatures[F_DEPTH])  featureVector.push_back( xmax - xmin );
            if(activeFeatures[F_HEIGHT]) featureVector.push_back( zmax - zmin );

            if(activeFeatures[F_ASPECT_RATIO_XY]) featureVector.push_back( (xmax - xmin) / (ymax - ymin) );
        }
    }
    #endif


    //
    // Added after ICRA'15 paper: HSV color features
    //
    #ifdef USE_HSV_FEATURES
    bool needHSVMeans = activeFeatures[F_HSV_MEAN_H] || activeFeatures[F_HSV_MEAN_S] || activeFeatures[F_HSV_MEAN_V];
    bool needHSVStdDevs = activeFeatures[F_HSV_STDDEV_H] || activeFeatures[F_HSV_STDDEV_S] || activeFeatures[F_HSV_STDDEV_V];
    if(needHSVMeans || needHSVStdDevs)
    {
        Eigen::Vector3d hsvMean = Eigen::Vector3d::Zero();
        std::vector<Eigen::Vector3d> hsvPoints; // for feature: HSV std devs
        
        foreach(int pointIndex, pointIndices) {
            const PointType& point = fullCloud.points[pointIndex];
            
            // Get RGB components
            unsigned char r = point.r;
            unsigned char g = point.g;
            unsigned char b = point.b;

            // Compute HSV vector
            int min;    // min. value of RGB
            int max;    // max. value of RGB
            int delta;  // delta RGB value in current sample

            if (r > g) { min = g; max = r; }
            else { min = r; max = g; }
            if (b > max) max = b;
            if (b < min) min = b;

            delta = max - min;
            float h = 0, s;
            float v = max;  

            if (delta == 0) { 
                h = 0; s = 0;
            }
            else {                                   
                s = delta / 255.0f;
                if ( r == max )         h = (      (g - b)/(float)delta) * 60.0;
                else if ( g == max )    h = ( 2 +  (b - r)/(float)delta) * 60.0;
                else if ( b == max )    h = ( 4 +  (r - g)/(float)delta) * 60.0;   
            }

            Eigen::Vector3d hsv;
            hsv(0) = h; hsv(1) = s * 100.0; hsv(2) = v * 100.0;                    

            // for feature: HSV means, HSV std devs
            hsvMean += hsv;

            // for feature: HSV std devs
            hsvPoints.push_back(hsv);
        }

        // for feature: HSV means
        if(activeFeatures[F_HSV_MEAN_H]) featureVector.push_back( hsvMean(0) );
        if(activeFeatures[F_HSV_MEAN_S]) featureVector.push_back( hsvMean(1) );
        if(activeFeatures[F_HSV_MEAN_V]) featureVector.push_back( hsvMean(2) );

        // for feature: HSV std devs
        if(needHSVStdDevs) {
            Eigen::Vector3d sumOfSquaredHSVDifferences = Eigen::Vector3d::Zero();
            foreach(const Eigen::Vector3d& hsv, hsvPoints) {
                Eigen::Vector3d meanDiff = hsv - hsvMean;
                Eigen::Vector3d meanDiffDotProduct = meanDiff.cwiseProduct(meanDiff);
                sumOfSquaredHSVDifferences += meanDiffDotProduct;
            }

            if(activeFeatures[F_HSV_STDDEV_H]) featureVector.push_back( sqrt(1/(pointCount - 1) * sumOfSquaredHSVDifferences(0)) );
            if(activeFeatures[F_HSV_STDDEV_S]) featureVector.push_back( sqrt(1/(pointCount - 1) * sumOfSquaredHSVDifferences(1)) );
            if(activeFeatures[F_HSV_STDDEV_V]) featureVector.push_back( sqrt(1/(pointCount - 1) * sumOfSquaredHSVDifferences(2)) );
        }
    } // end HSV color features
    #endif


    //
    // Added after ICRA'15 paper: YCbCr features
    //
    #ifdef USE_YCBCR_FEATURES
    bool needYCbCrMeans = activeFeatures[F_YCBCR_MEAN_Y] || activeFeatures[F_YCBCR_MEAN_CB] || activeFeatures[F_YCBCR_MEAN_CR];
    bool needYCbCrStdDevs = activeFeatures[F_YCBCR_STDDEV_Y] || activeFeatures[F_YCBCR_STDDEV_CB] || activeFeatures[F_YCBCR_STDDEV_CR];
    if(needYCbCrMeans || needYCbCrStdDevs)
    {
        Eigen::Vector3d ycbcrMean = Eigen::Vector3d::Zero();
        std::vector<Eigen::Vector3d> ycbcrPoints; // for feature: YCbCr std devs
        
        foreach(int pointIndex, pointIndices) {
            const PointType& point = fullCloud.points[pointIndex];
            
            // Get RGB components
            unsigned char r = point.r;
            unsigned char g = point.g;
            unsigned char b = point.b;

            // Compute YCbCr vector
            double y  =  0.299   * r + 0.587   * g + 0.114   * b;
            double cb = -0.16874 * r - 0.33126 * g + 0.50000 * b;
            double cr =  0.50000 * r - 0.41869 * g - 0.08131 * b;

            Eigen::Vector3d ycbcr;
            ycbcr(0) = y; ycbcr(1) = cb; ycbcr(2) = cr;                    

            // for feature: YCbCr means, YCbCr std devs
            ycbcrMean += ycbcr;

            // for feature: YCbCr std devs
            ycbcrPoints.push_back(ycbcr);
        }

        // for feature: YCbCr means
        if(activeFeatures[F_YCBCR_MEAN_Y])  featureVector.push_back( ycbcrMean(0) );
        if(activeFeatures[F_YCBCR_MEAN_CB]) featureVector.push_back( ycbcrMean(1) );
        if(activeFeatures[F_YCBCR_MEAN_CR]) featureVector.push_back( ycbcrMean(2) );

        // for feature: YCbCr std devs
        if(needYCbCrStdDevs) {
            Eigen::Vector3d sumOfSquaredYCbCrDifferences = Eigen::Vector3d::Zero();
            foreach(const Eigen::Vector3d& ycbcr, ycbcrPoints) {
                Eigen::Vector3d meanDiff = ycbcr - ycbcrMean;
                Eigen::Vector3d meanDiffDotProduct = meanDiff.cwiseProduct(meanDiff);
                sumOfSquaredYCbCrDifferences += meanDiffDotProduct;
            }

            if(activeFeatures[F_YCBCR_STDDEV_Y])  featureVector.push_back( sqrt(1/(pointCount - 1) * sumOfSquaredYCbCrDifferences(0)) );
            if(activeFeatures[F_YCBCR_STDDEV_CB]) featureVector.push_back( sqrt(1/(pointCount - 1) * sumOfSquaredYCbCrDifferences(1)) );
            if(activeFeatures[F_YCBCR_STDDEV_CR]) featureVector.push_back( sqrt(1/(pointCount - 1) * sumOfSquaredYCbCrDifferences(2)) );
          }  
    } // end YCbCr color features
    #endif


    //
    // Added after ICRA'15 paper: RGB color features
    //
    #ifdef USE_RGB_FEATURES
    bool needRGBMeans = activeFeatures[F_RGB_MEAN_R] || activeFeatures[F_RGB_MEAN_G] || activeFeatures[F_RGB_MEAN_B];
    bool needRGBStdDevs = activeFeatures[F_RGB_STDDEV_R] || activeFeatures[F_RGB_STDDEV_G] || activeFeatures[F_RGB_STDDEV_B];
    if(needRGBMeans || needRGBStdDevs)
    {
        Eigen::Vector3d rgbMean = Eigen::Vector3d::Zero();
        std::vector<Eigen::Vector3d> rgbPoints; // for feature: RGB std devs
        
        foreach(int pointIndex, pointIndices) {
            const PointType& point = fullCloud.points[pointIndex];
            
            // Get RGB components
            Eigen::Vector3d rgb(point.r, point.g, point.b);

            // for feature: RGB means, RGB std devs
            rgbMean += rgb;

            // for feature: RGB std devs
            rgbPoints.push_back(rgb);
        }

        // for feature: RGB means
        if(activeFeatures[F_RGB_MEAN_R]) featureVector.push_back( rgbMean(0) );
        if(activeFeatures[F_RGB_MEAN_G]) featureVector.push_back( rgbMean(1) );
        if(activeFeatures[F_RGB_MEAN_B]) featureVector.push_back( rgbMean(2) );

        // for feature: RGB std devs
        if(needRGBStdDevs) {
            Eigen::Vector3d sumOfSquaredRGBDifferences = Eigen::Vector3d::Zero();
            foreach(const Eigen::Vector3d& rgb, rgbPoints) {
                Eigen::Vector3d meanDiff = rgb - rgbMean;
                Eigen::Vector3d meanDiffDotProduct = meanDiff.cwiseProduct(meanDiff);
                sumOfSquaredRGBDifferences += meanDiffDotProduct;
            }

            if(activeFeatures[F_RGB_STDDEV_R]) featureVector.push_back( sqrt(1/(pointCount - 1) * sumOfSquaredRGBDifferences(0)) );
            if(activeFeatures[F_RGB_STDDEV_G]) featureVector.push_back( sqrt(1/(pointCount - 1) * sumOfSquaredRGBDifferences(1)) );
            if(activeFeatures[F_RGB_STDDEV_B]) featureVector.push_back( sqrt(1/(pointCount - 1) * sumOfSquaredRGBDifferences(2)) );
        }
    } // end RGB color features
    #endif


    // Safety check that we got the correct number of features
    #ifdef _DEBUG
        size_t expectedFeatureVectorLength = 0;
        for(size_t i = 0; i < activeFeatures.size(); i++) expectedFeatureVectorLength += activeFeatures[i] ? 1 : 0;
        ROS_ASSERT(featureVector.size() == expectedFeatureVectorLength);
    #endif
}


size_t FeatureCalculator::getFeatureCount() const {
    return F_COUNT;
} 