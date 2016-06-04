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

    float m_minVoxelSize, m_maxBorderWithoutVoxel, m_zOffset;
    Volume m_parentVolume;
    pcl::PointXYZ m_parentVolumeSize;
    std::vector<pcl::PointXYZ> m_voxelAspectRatios;
    std::vector<float> m_voxelSizeIncrements;
    bool m_overlapEnabled;
};


#endif // TESSELLATION_GENERATOR_H_