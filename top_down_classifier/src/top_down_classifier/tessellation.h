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