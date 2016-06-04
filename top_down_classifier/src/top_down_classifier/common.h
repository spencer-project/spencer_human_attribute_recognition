#ifndef COMMON_H_
#define COMMON_H_


#include <vector>
#include <boost/shared_ptr.hpp>
#include <pcl/common/common.h>


typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> PointCloud;

/// A class label (for a binary classifier, usually 0 or 1)
typedef int class_label;


#endif // COMMON_H_