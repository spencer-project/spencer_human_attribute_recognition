#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <cstdio>

#include "../features.h"
#include "../volume.h"

using namespace std;

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void sampleSphere(PointCloud::Ptr& cloud, const Eigen::Vector3d& origin, double radius, int pointCount)
{
    boost::mt19937 rng;
    boost::normal_distribution<> nd(0.0, 1.0);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

    for(int i = 0; i < pointCount; i++) {
        Eigen::Vector3d v(var_nor(), var_nor(), var_nor());
        Eigen::Vector3d pointOnSphere = radius / v.norm() * v;
        cloud->points.push_back( PointType(pointOnSphere(0), pointOnSphere(1), pointOnSphere(2)) );
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void samplePlane(PointCloud::Ptr& cloud, const Eigen::Vector3d& normalVector, double distanceToOrigin, int pointCount, double scale1 = 1.0, double scale2 = 1.0)
{
    boost::mt19937 rng;
    boost::variate_generator<boost::mt19937&, boost::uniform_real<> > uni1(rng, boost::uniform_real<>(-0.2 * scale1, +0.2 * scale1));
    boost::variate_generator<boost::mt19937&, boost::uniform_real<> > uni2(rng, boost::uniform_real<>(-0.2 * scale2, +0.2 * scale2));

    for(int i = 0; i < pointCount; i++) {
        double x0 = uni1();
        double x1 = uni2();
        double x2 = (-distanceToOrigin - normalVector(0)*x0 - normalVector(1)*x1) / normalVector(2);

        Eigen::Vector3d pointOnPlane(x0, x1, x2);
        cloud->points.push_back( PointType(pointOnPlane(0), pointOnPlane(1), pointOnPlane(2)) );
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_features");
    ros::NodeHandle nodeHandle("");
    ros::NodeHandle privateHandle("~");

    // Create feature calculator
    FeatureCalculator featureCalculator;
    const std::vector<string>& featureNames = featureCalculator.getFeatureNames();

    pcl::PointXYZ minCoords(-0.3, -0.3, 0), maxCoords(0.3, 0.3, 1.8);
    Volume parentVolume(minCoords, maxCoords);

    // Create point cloud publisher
    ros::Publisher pointCloudPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("cloud", 1, true);
        

    for(int testCase = 0; testCase < 7 && ros::ok(); testCase++) {
        PointCloud::Ptr testCloud(new PointCloud);
        testCloud->header.stamp = ros::Time::now().toNSec() / 1000;
        testCloud->header.frame_id = "extracted_cloud_frame";

        switch(testCase)
        {
            case 0:
            {
                ROS_INFO("=== SMALL SPHERE ===");
                Eigen::Vector3d origin(1, 2, 3);
                sampleSphere(testCloud, origin, 0.2, 100);
                break;
            }
            case 1:
            {
                ROS_INFO("=== LARGE SPHERE ===");
                Eigen::Vector3d origin(0, 0, 0);
                sampleSphere(testCloud, origin, 0.4, 500);
                break;
            }
            case 2:
            {
                ROS_INFO("=== EXTRA-LARGE SPHERE ===");
                Eigen::Vector3d origin(1, 1, 0);
                sampleSphere(testCloud, origin, 0.8, 500);
                break;
            }

            case 3:
            {
                ROS_INFO("=== FIRST PLANE ===");
                Eigen::Vector3d normalVector(0.9, 0, 0.3);
                samplePlane(testCloud, normalVector.normalized(), 0.1, 100);
                break;
            }
            case 4:
            {
                ROS_INFO("=== SECOND PLANE ===");
                Eigen::Vector3d normalVector(0.2, 0, 0.5);
                samplePlane(testCloud, normalVector.normalized(), -1.0, 500);
                break; 
            }
            case 5:
            {
                ROS_INFO("=== THIRD PLANE ===");
                Eigen::Vector3d normalVector(0, 0, 1);
                samplePlane(testCloud, normalVector.normalized(), 0.0, 200);
                break; 
            }
            case 6:
            {
                ROS_INFO("=== FOURTH PLANE ===");
                Eigen::Vector3d normalVector(0, 0, 1);
                samplePlane(testCloud, normalVector.normalized(), 0.0, 200, 0.1, 10.0);
                break; 
            }
        }

        std::vector<int> indices;
        for(int i = 0; i < testCloud->points.size(); i++) indices.push_back(i);

        std::vector<double> featureVector;
        featureCalculator.calculateFeatures(parentVolume, *testCloud, indices, featureCalculator.maskAllFeaturesActive(), featureVector);

        for(int f = 0; f < featureVector.size(); f++) {
            ROS_INFO("Feature %s:  %g", featureNames[f].c_str(), featureVector[f]);
        }


        pointCloudPublisher.publish(testCloud);
        ros::spinOnce();
        
        ROS_INFO("Press ENTER to show next test case...");
        getchar();
    }

    return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
