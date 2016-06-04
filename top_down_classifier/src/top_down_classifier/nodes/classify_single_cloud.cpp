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
#include <boost/algorithm/string/predicate.hpp>
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
    class_label label = g_classifier.classify(personCloud, &weightedSum); // invoke classifier
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

        std::stringstream ss;
        ss << "Predicted class label is " << label;

        if(boost::algorithm::contains(g_classifier.getCategory(), "gender")) {
            ss << " (" << (label == 1 ? "MALE" : "FEMALE") << ")";
        }
        else {
            ss << " (" << (label == 1 ? "POSITIVE" : "NEGATIVE") << ")";
        }

        ROS_INFO_STREAM(ss.str());
        return label;
    }
    else {
        ROS_ERROR("_filename argument (pointing to a PCD file with extracted person cloud) has not been specified!");
        return -1;
    }
        
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
