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