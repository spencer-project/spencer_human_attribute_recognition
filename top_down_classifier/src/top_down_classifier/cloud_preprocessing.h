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
