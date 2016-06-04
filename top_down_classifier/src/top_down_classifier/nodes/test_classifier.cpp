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
#include "../volume_visualizer.h"
#include "../classifier.h"
#include "../features.h"

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


// Global variables
ros::Publisher g_pointCloudPublisher;
TopDownClassifier g_classifier;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double testClassifier(const std::string& listFilename)
{
    std::ifstream listFile(listFilename.c_str());
    if(!listFile.good()) {
        ROS_ERROR_STREAM("Failed to open file " << listFilename);
        exit(EXIT_FAILURE);
    }

    string cloudFilename;
    class_label groundtruthLabel;
    size_t numCloudsTotal = 0, numCloudsCorrectlyClassified = 0;
    
    // Skip comments at beginning of file
    std::string commentString;
    const size_t numCommentLinesAtBeginning = 2;
    for(size_t i = 0; i < numCommentLinesAtBeginning; i++) std::getline(listFile, commentString);

    // Process cloud by cloud individually
    ros::WallTime startTime = ros::WallTime::now();
    ros::WallDuration accumulatedDurationWithoutIO(0);
    while (listFile >> cloudFilename >> groundtruthLabel)
    {
        // Load point cloud from disk
        PointCloud::Ptr personCloud(new PointCloud);
        string filename = cloudFilename;
        ROS_DEBUG_STREAM("Loading " << filename << "...");

        if(pcl::io::loadPCDFile<PointType>(filename, *personCloud) == -1)
        {
            ROS_ERROR("Couldn't read file %s\n", filename.c_str());
            return (-1);
        }
        numCloudsTotal++;

        // Publish point cloud
        if(g_pointCloudPublisher.getNumSubscribers() > 0) {
            personCloud->header.frame_id = "extracted_cloud_frame";
            personCloud->header.stamp = ros::Time::now().toNSec() / 1000;
            g_pointCloudPublisher.publish(personCloud);
        }

        // Classify cloud
        ros::WallTime startTimeWithoutIO = ros::WallTime::now();
        
        class_label label = g_classifier.classify(personCloud); // invoke classifier
        if(label == groundtruthLabel) numCloudsCorrectlyClassified++;

        ros::WallTime endTimeWithoutIO = ros::WallTime::now();
        accumulatedDurationWithoutIO += endTimeWithoutIO - startTimeWithoutIO;
    }

    ros::WallTime endTime = ros::WallTime::now();

    double duration = (endTime - startTime).toSec();
    ROS_INFO("Testing of %d person clouds (including loading from disk) took in total %.2f sec, that's on average %.1f ms per frame or %.1f Hz!",
        (int) numCloudsTotal, duration, 1000.0 * duration / numCloudsTotal, numCloudsTotal / duration);

    duration = accumulatedDurationWithoutIO.toSec();
    ROS_INFO("Accumulated testing time WITHOUT loading from disk (this is an approximate estimate!): %.2f sec, that's on average %.1f ms per frame or %.1f Hz!",
        duration, 1000.0 * duration / numCloudsTotal, numCloudsTotal / duration);

    // Return accuracy on test set
    return (double) numCloudsCorrectlyClassified / numCloudsTotal;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_top_down_classifier");
    ros::NodeHandle nodeHandle("");
    ros::NodeHandle privateHandle("~");

    //
    // Parse arguments
    //

    bool showBestTessellation; std::string listFilename, modelFilename; int numThreads, numberOfVolumesToShow;

    privateHandle.param<bool>("show_best_tessellation", showBestTessellation, true);
    privateHandle.param<int>("num_volumes_to_show", numberOfVolumesToShow, 50);
    privateHandle.param<std::string>("list_file", listFilename, "");
    privateHandle.param<std::string>("model", modelFilename, "");
    privateHandle.param<int>("num_threads", numThreads, 5);

    omp_set_num_threads(numThreads);
    ROS_INFO_STREAM("Using " << numThreads << " parallel threads for feature computations!");
    
    // Create point cloud publisher
    g_pointCloudPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("cloud", 1, true);
    
    
    //
    // Load classifier
    //

    if(modelFilename.empty()) {
        ROS_ERROR_STREAM("The _model argument was not specified; this is required, and must point to a YAML file containing the learned classifier.");
        exit(EXIT_FAILURE);
    }
    g_classifier.init(modelFilename);    


    //
    // Visualize parent volume and best tessellation (latch enabled, so only need to publish once)
    //

    ROS_INFO("Visualizing learned best tessellation and parent volume...");

    VolumeVisualizer visualizer;
    visualizer.clear();
    visualizer.visualize(g_classifier.getParentVolume(), 0, "Parent Volume", 0xffff00, 3.0f);

    if(g_classifier.isOptimized())
    {
        // Show best tessellation
        const std::vector<Volume>& volumes = g_classifier.getVolumes();
        const std::vector<size_t>& bestVolumeIndices = g_classifier.getIndicesOfBestVolumes();
        for(size_t i = 0; i < std::min((int)bestVolumeIndices.size(), numberOfVolumesToShow); i++) {
            const size_t volumeIndex = bestVolumeIndices[i];
            const Volume& volume = volumes[volumeIndex];
            //ROS_INFO_STREAM("Volume: " << volume);
            visualizer.visualize(volume, 1+i, "Learned best tessellation"); 
        }
        ROS_INFO_STREAM("" << volumes.size() + 1 << " volume(s) have been visualized!");
    }
    else {
        // Skip showing best tessellation, since we don't know which tessellation volumes belong to it, and their qualities
        ROS_WARN_STREAM("Classifier is not optimized, skipping visualization of all " << g_classifier.getVolumes().size() << " tessellation volumes");
    }

    ros::WallRate wallRate(30);
    
    // Give subscribers enough time to subscribe, since we only publish once and might exit thereafter
    for(size_t i = 0; i < 60; i++) { wallRate.sleep(); ros::spinOnce(); }
    visualizer.publish();
    for(size_t i = 0; i < 60; i++) { wallRate.sleep(); ros::spinOnce(); }


    //
    // Test classifier
    //

    if(!listFilename.empty()) {
        // Test classifier on provided list file (each line contains a cloud filename + label
        // separated by space or tabulator; first 2 lines are ignored)
        ROS_INFO_STREAM("Testing classifier on " << listFilename);
        double resultingAccuracy = testClassifier(listFilename); 
        ROS_INFO("");  
        ROS_INFO("Testing complete, average accuracy is %.2f%%!", 100.0 * resultingAccuracy);
    }
    else {
        ROS_WARN("_list_file argument was not specified; not testing on any point clouds. This file should contain "
                 "a cloud (PCD) filename + label separated by space or tabulator per line; the first two lines are ignored.");
    }
        
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
