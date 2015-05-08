#ifndef VOLUME_H_
#define VOLUME_H_

#include "common.h"


/// An axis-aligned sub-volume of a tessellation.
class Volume {
public:
    Volume(const pcl::PointXYZ& minCoordinates, const pcl::PointXYZ& maxCoordinates);
    Volume() {}

    /// Initializes a volume using the bounding box of the given point cloud.
    static Volume fromCloudBBox(const PointCloud& cloud);

    /// Initializes a volume as the bounding volume of the given sub-volumes.
    static Volume getBoundingVolume(const std::vector<Volume>& subVolumes);

    /// @arg inputCloud The full point cloud to extract a sub-volume from
    /// @arg extractedCloud Can be NULL if not of interst
    /// @arg extractedIndices Can be NULL if not of interest
    void getPointsInsideVolume(const PointCloud& inputCloud, PointCloud::Ptr extractedCloud, std::vector<int>* extractedIndices) const;

    /// Return minimum x,y,z coordinates
    const pcl::PointXYZ& getMinCoordinates() const { return m_minCoordinates; }

    /// Return maximum x,y,z coordinates
    const pcl::PointXYZ& getMaxCoordinates() const { return m_maxCoordinates; }

    /// Return minimum x,y,z coordinates
    pcl::PointXYZ& getMinCoordinates() { return m_minCoordinates; }

    /// Return maximum x,y,z coordinates
    pcl::PointXYZ& getMaxCoordinates() { return m_maxCoordinates; }

    /// Get center of volume
    pcl::PointXYZ getCenter() const;

    /// Get size of volume
    pcl::PointXYZ getSize() const;
 
private:
    pcl::PointXYZ m_minCoordinates, m_maxCoordinates;
};


/// For dumping volume coordinates into an output stream
std::ostream& operator<< (std::ostream &os, const Volume& volume);



#endif // VOLUME_H_