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

#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>

#include "common.h"


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void scaleAndCropCloudToTargetSize(PointCloud::Ptr personCloud, double scaleZto, double cropZmin, double cropZmax)
{
    // Zero or negative crop_z_max indicates that we shall crop relative to the top of the cloud
    // (i.e. crop_z_min = -0.3, crop_z_max = 0.0 means we shall leave only the topmost 30 cm of the cloud)
    bool cropRelativeToTopOfCloud = cropZmax <= 0;

    // Get minima/maxima of cloud
    PointType minCoords, maxCoords;
    if(scaleZto > 0 || cropRelativeToTopOfCloud) {
        // Find maximum xyz values of cloud and then scale accordingly
        pcl::getMinMax3D(*personCloud, minCoords, maxCoords);
    }

    //
    // Scale cloud
    //
    if(scaleZto > 0) {
        double zscale = scaleZto / maxCoords.z;
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.scale(Eigen::Vector3f(1.f, 1.f, zscale));
        pcl::transformPointCloud (*personCloud, *personCloud, transform);

        minCoords.z *= zscale;
        maxCoords.z *= zscale;
    }

    // Make coordinates relative to the top of the cloud if required
    if(cropRelativeToTopOfCloud) {
        cropZmin = maxCoords.z + cropZmin; // cropZmin assumed to be negative
        cropZmax = maxCoords.z + cropZmax; // cropZmax zero or negative
    }

    //
    // Crop cloud
    //
    if(std::isfinite(cropZmin) || std::isfinite(cropZmax)) {
        pcl::PassThrough<PointType> pass;
        pass.setInputCloud (personCloud);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (cropZmin, cropZmax);
        pass.filter(*personCloud);
    }

    // If we cropped relative to the top but did not scale, translate into origin
    if(scaleZto <= 0 && cropRelativeToTopOfCloud) {
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translate(Eigen::Vector3f(0.f, 0.f, -cropZmin));
        pcl::transformPointCloud (*personCloud, *personCloud, transform);
    }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
