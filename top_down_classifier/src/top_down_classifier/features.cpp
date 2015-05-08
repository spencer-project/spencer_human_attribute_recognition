#include "features.h"

#include <Eigen/Eigenvalues> 

#include <algorithm>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

enum Features {
    // NOTE: New features must be added to the end of the list, otherwise
    // existing learned classifier models will break!
    F_NUM_POINTS,
    F_SPHERICITY,
    F_FLATNESS,
    F_LINEARITY,
    F_STDDEV,
    F_KURTOSIS,
    F_MAD_MEDIAN,
    F_NORM_RES_PLANARITY,
    F_NUM_POINTS_RATIO,

    F_COUNT, /* add new features before this element */
};


FeatureCalculator::FeatureCalculator() 
{
    m_featureNames.resize(F_COUNT);
    m_allFeaturesMask = std::vector<bool>(F_COUNT, true);

    m_featureNames[F_NUM_POINTS]            = "Number of points";
    m_featureNames[F_SPHERICITY]            = "Sphericity";
    m_featureNames[F_FLATNESS]              = "Flatness";
    m_featureNames[F_LINEARITY]             = "Linearity";
    m_featureNames[F_STDDEV]                = "Standard deviation w.r.t. centroid";
    m_featureNames[F_KURTOSIS]              = "Kurtosis w.r.t. centroid";
    m_featureNames[F_MAD_MEDIAN]            = "Average deviation from median";
    m_featureNames[F_NORM_RES_PLANARITY]    = "Normalized residual planarity";
    m_featureNames[F_NUM_POINTS_RATIO]      = "Number of points ratio relative to total point count in cloud";  
}

// NOTE: This code might be a bit hard to read since it contains optimizations to compute only what is absolutely
// required, given the bit mask of active features for this particular volume.
void FeatureCalculator::calculateFeatures(const Volume& parentVolume, const PointCloud& fullCloud, const std::vector<int>& pointIndices,
    const std::vector<bool>& activeFeatures, std::vector<double>& featureVector) const
{
    // Feature: Number of points
    double pointCount = pointIndices.size();
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
            const pcl::PointXYZ& point = fullCloud.points[pointIndex];
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
            const pcl::PointXYZ& point = fullCloud.points[pointIndex];
            sampleMean += Eigen::Vector3d(point.x, point.y, point.z);
        }
        sampleMean /= pointCount;
    }
    else if(!needSampleMean && needMedian) {
        xs.reserve(pointIndices.size());
        ys.reserve(pointIndices.size());
        zs.reserve(pointIndices.size());

        foreach(int pointIndex, pointIndices) {
            const pcl::PointXYZ& point = fullCloud.points[pointIndex];
            
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
            const pcl::PointXYZ& point = fullCloud.points[pointIndex];
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
            const pcl::PointXYZ& point = fullCloud.points[pointIndex];
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