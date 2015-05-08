#ifndef TESSELLATION_GENERATOR_H_
#define TESSELLATION_GENERATOR_H_

#include "volume.h"
#include "tessellation.h"
#include "common.h"

#include <list>


/// Generates an axis-aligned volume tessellation of a point cloud as described in the paper
/// "Tracking People in 3D Using a Bottom-Up Top-Down Detector" by Spinello et al., ICRA 2011
/// using the top-down method
class TessellationGenerator {
public:
    TessellationGenerator(const Volume& parentVolume, std::vector<pcl::PointXYZ> voxelAspectRatios, std::vector<float> voxelSizeIncrements, float minVoxelSize, bool overlapEnabled = true);

    /// Generates all possible tessellations under the given parameters.
    void generateTessellations(std::list<Tessellation>& tessellations);

private:
    /// Generate a single tessellation at a single scale and aspect ratio.
    Tessellation generateTessellation(const pcl::PointXYZ& voxelSize);
    Tessellation generateOverlappingTessellation(const pcl::PointXYZ& voxelSize);

    void centerVolumes(std::vector<Volume>& volumes);

    float m_minVoxelSize, m_maxBorderWithoutVoxel;
    Volume m_parentVolume;
    pcl::PointXYZ m_parentVolumeSize;
    std::vector<pcl::PointXYZ> m_voxelAspectRatios;
    std::vector<float> m_voxelSizeIncrements;
    bool m_overlapEnabled;
};


#endif // TESSELLATION_GENERATOR_H_