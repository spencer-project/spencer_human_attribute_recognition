#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <string>
#include <cstdio>
#include <fstream>
#include <sstream>

#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ml/ml.hpp>

#include "../volume.h"
#include "../classifier.h"
#include "../features.h"

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


// Global variables
ros::Publisher g_pointCloudPublisher;
bool g_publishCloud;
TopDownClassifier g_classifier;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class_label classifySingleCloud(const std::string& cloudFilename)
{  
    // Load point cloud from disk
    PointCloud::Ptr personCloud(new PointCloud);
    ROS_DEBUG_STREAM("Loading " << cloudFilename << "...");

    if(pcl::io::loadPCDFile<PointType>(cloudFilename, *personCloud) == -1)
    {
        ROS_ERROR("Couldn't read file %s\n", cloudFilename.c_str());
        exit(-1);
    }
    
    // Publish point cloud
    if(g_publishCloud) {
        personCloud->header.frame_id = "extracted_cloud_frame";
        personCloud->header.stamp = ros::Time::now().toNSec() / 1000;
        g_pointCloudPublisher.publish(personCloud);
        ros::WallRate rate(25);
        for(int i = 0; i < 10; i++) { ros::spinOnce(); rate.sleep(); }
    }

    // Classify cloud
    ros::WallTime startTimeWithoutIO = ros::WallTime::now();
    double weightedSum = std::numeric_limits<double>::quiet_NaN();
    class_label label = g_classifier.classify(*personCloud, &weightedSum); // invoke classifier
    ros::WallDuration accumulatedDurationWithoutIO = ros::WallTime::now() - startTimeWithoutIO;
    
    double duration = accumulatedDurationWithoutIO.toSec();
    ROS_INFO("Classification time WITHOUT loading from disk: %.2f sec, that's on average %.1f ms per frame or %.1f Hz!",
        duration, 1000.0 * duration, 1.0 / duration);

    ROS_INFO_STREAM("Weighted sum returned by strong classifier is " << std::fixed << std::setprecision(5) << weightedSum);

    return label;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "classify_single_cloud");
    ros::NodeHandle nodeHandle("");
    ros::NodeHandle privateHandle("~");

    //
    // Parse arguments
    //

    bool showBestTessellation; std::string cloudFilename, modelFilename; int numThreads;
    privateHandle.param<std::string>("filename", cloudFilename, "");
    privateHandle.param<std::string>("model", modelFilename, "");
    privateHandle.param<int>("num_threads", numThreads, 5);
    privateHandle.param<bool>("publish_cloud", g_publishCloud, false);

    omp_set_num_threads(numThreads);
    ROS_INFO_STREAM("Using " << numThreads << " parallel threads for feature computations!");
    
    // Create point cloud publisher
    g_pointCloudPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("cloud", 1, true);
    
    
    //
    // Load classifier
    //

    if(modelFilename.empty()) {
        ROS_ERROR_STREAM("The _model argument was not specified; this is required, and must point to a YAML file containing the learned classifier.");
        return -1;
    }
    g_classifier.init(modelFilename);    


    //
    // Classify cloud
    //

    if(!cloudFilename.empty()) {
        // Test classifier on provided list file (each line contains a cloud filename + label
        // separated by space or tabulator; first 2 lines are ignored)
        class_label label = classifySingleCloud(cloudFilename); 
        ROS_INFO_STREAM("Predicted class label is " << label << " (" << (label == 1 ? "MALE" : "FEMALE") << ")!");
        return label;
    }
    else {
        ROS_ERROR("_filename argument (pointing to a PCD file with extracted person cloud) has not been specified!");
        return -1;
    }
        
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
