#include "volume.h"

#include <pcl/common/common.h>

#include <iomanip>
#include <cassert>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


Volume::Volume(const pcl::PointXYZ& minCoordinates, const pcl::PointXYZ& maxCoordinates)
{
    assert(minCoordinates.x < maxCoordinates.x);
    assert(minCoordinates.y < maxCoordinates.y);
    assert(minCoordinates.z < maxCoordinates.z);   

    m_minCoordinates = minCoordinates;
    m_maxCoordinates = maxCoordinates;
}


Volume Volume::fromCloudBBox(const PointCloud& cloud)
{
    PointType minCoordinates, maxCoordinates;
    pcl::getMinMax3D(cloud, minCoordinates, maxCoordinates);

    pcl::PointXYZ minCoordinates2, maxCoordinates2;

    minCoordinates2.x = minCoordinates.x;
    minCoordinates2.y = minCoordinates.y;
    minCoordinates2.z = minCoordinates.z;
 
    maxCoordinates2.x = maxCoordinates.x;
    maxCoordinates2.y = maxCoordinates.y;
    maxCoordinates2.z = maxCoordinates.z;      

    return Volume(minCoordinates2, maxCoordinates2);
}


Volume Volume::getBoundingVolume(const std::vector<Volume>& subVolumes)
{
    pcl::PointXYZ minCoords, maxCoords;

    foreach(const Volume& volume, subVolumes) {
        minCoords.x = std::min(minCoords.x, volume.getMinCoordinates().x);
        minCoords.y = std::min(minCoords.y, volume.getMinCoordinates().y);
        minCoords.z = std::min(minCoords.z, volume.getMinCoordinates().z);
     
        maxCoords.x = std::max(maxCoords.x, volume.getMaxCoordinates().x);
        maxCoords.y = std::max(maxCoords.y, volume.getMaxCoordinates().y);
        maxCoords.z = std::max(maxCoords.z, volume.getMaxCoordinates().z);      
    }

    return Volume(minCoords, maxCoords);
}


void Volume::getPointsInsideVolume(const PointCloud& inputCloud, PointCloud::Ptr extractedCloud, std::vector<int>* extractedIndices) const
{
    std::vector<int> indicesInVolume;

    for(size_t i = 0; i < inputCloud.points.size(); i++) {
        const PointType& currentPoint = inputCloud.points[i];

        if(currentPoint.x < m_minCoordinates.x || currentPoint.x > m_maxCoordinates.x
        || currentPoint.y < m_minCoordinates.y || currentPoint.y > m_maxCoordinates.y
        || currentPoint.z < m_minCoordinates.z || currentPoint.z > m_maxCoordinates.z)
        {
            // Point is outside of volume.
        }
        else {
            // Point is inside of volume
            if(indicesInVolume.size() == indicesInVolume.capacity()) {
                indicesInVolume.reserve(indicesInVolume.size() + 1000); // trying to reduce number of repeated memory allocations
            }
            indicesInVolume.push_back(i);
        }
    }

    if(extractedCloud) {
        *extractedCloud = PointCloud(inputCloud, indicesInVolume);
    }
    if(extractedIndices) {
        *extractedIndices = indicesInVolume;
    }
}


pcl::PointXYZ Volume::getCenter() const {
    pcl::PointXYZ center;
    center.x = (m_minCoordinates.x + m_maxCoordinates.x) / 2.0f;
    center.y = (m_minCoordinates.y + m_maxCoordinates.y) / 2.0f;
    center.z = (m_minCoordinates.z + m_maxCoordinates.z) / 2.0f;
    return center;
}


pcl::PointXYZ Volume::getSize() const {
    pcl::PointXYZ size;
    size.x = m_maxCoordinates.x - m_minCoordinates.x;
    size.y = m_maxCoordinates.y - m_minCoordinates.y;
    size.z = m_maxCoordinates.z - m_minCoordinates.z;

    return size;
}


std::ostream& operator<< (std::ostream &os, const Volume& volume) {
    const pcl::PointXYZ& min = volume.getMinCoordinates();
    const pcl::PointXYZ& max = volume.getMinCoordinates();
    const pcl::PointXYZ& c = volume.getCenter();
    const pcl::PointXYZ& s = volume.getCenter();

    os << std::fixed << std::setprecision(2)
       << "(" << min.x << ", " << min.y << ", " << min.z << ") -- "
       << "(" << max.x << ", " << max.y << ", " << max.z << "), c="
       << "(" <<   c.x << ", " <<   c.y << ", " <<   c.z << "), s="
       << "(" <<   s.x << " x "<<   s.y << " x "<<   s.z << ")";
}