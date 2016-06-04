#ifndef TESSELLATION_H_
#define TESSELLATION_H_

#include "volume.h"
#include "common.h"

/// Represents a regular, axis-aligned face-to-face tessellation of a volume B
class Tessellation {
public:
    /// A tessellation of a given parent volume B by using equally sized voxels of size voxelSize.
    Tessellation(const pcl::PointXYZ& voxelSize, std::vector<Volume>& volumes, bool isOverlapping = false) {
        m_voxelSize = voxelSize;
        m_volumes = volumes;
        m_isOverlapping = isOverlapping;
    }

    /// Returns the volumes belonging to this tesellation (=sub-volumes of the parent volume). 
    const std::vector<Volume>& getVolumes() const {
        return m_volumes;
    }

    const pcl::PointXYZ& getVoxelSize() const {
        return m_voxelSize;
    }

    const bool isOverlapping() const {
        return m_isOverlapping;
    }

    // Used in pre-training to delete volumes which are mostly empty / produce a lot of NaN feature values.
    void deleteVolume(size_t index) {
        m_volumes.erase(m_volumes.begin() + index);
    }


private:
    /// The width, depth and height of all voxels in this particular (regular) tessellation.
    pcl::PointXYZ m_voxelSize; // = w, d, h

    std::vector<Volume> m_volumes;
    bool m_isOverlapping;
};


#endif // TESSELLATION_H_