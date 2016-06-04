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

#include "volume_visualizer.h"

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


VolumeVisualizer::VolumeVisualizer()
{
    ros::NodeHandle nh;
    m_markerArrayPublisher.reset(new ros::Publisher( nh.advertise<visualization_msgs::MarkerArray>("tessellation_volumes", 100, true) ));

    ros::NodeHandle pnh("~");
    pnh.param("marker_frame", m_markerFrame, std::string("extracted_cloud_frame"));
    pnh.param("color_scheme", m_colorScheme, std::string("rainbow"));
    pnh.param("alpha", m_alpha, 0.3);
    pnh.param("line_width", m_lineWidth, 0.005);
    pnh.param("wireframe", m_showWireframe, true);
    pnh.param("solid", m_showSolid, true);
}


void VolumeVisualizer::visualize(const Volume& volume, int paletteColorIndex, const std::string& ns, unsigned int borderColor, float borderWidth)
{
    pcl::PointXYZ center = volume.getCenter(),  size = volume.getSize();
    pcl::PointXYZ minCoords = volume.getMinCoordinates(), maxCoords = volume.getMaxCoordinates();

    ros::Time now = ros::Time::now();

    if(m_showSolid) {
        visualization_msgs::Marker boxMarker;

        boxMarker.header.frame_id = m_markerFrame;
        boxMarker.header.stamp = now;
        boxMarker.type = visualization_msgs::Marker::CUBE;
        boxMarker.color = getPaletteColor(paletteColorIndex);
        boxMarker.pose.position.x = center.x;
        boxMarker.pose.position.y = center.y;
        boxMarker.pose.position.z = center.z;
        boxMarker.scale.x = size.x;
        boxMarker.scale.y = size.y;
        boxMarker.scale.z = size.z;
        boxMarker.ns = ns;
        boxMarker.id = m_oldMarkerIds.size();

        m_oldMarkerIds.push_back( std::make_pair<std::string, int>(boxMarker.ns, boxMarker.id) );
        m_markerArray.markers.push_back(boxMarker);
    }

    if(m_showWireframe) {
        visualization_msgs::Marker lineListMarker;

        lineListMarker.header.frame_id = m_markerFrame;
        lineListMarker.header.stamp = now;
        lineListMarker.type = visualization_msgs::Marker::LINE_LIST;
        lineListMarker.color = colorFromInt(borderColor);

        lineListMarker.ns = ns + " Wireframe";
        lineListMarker.id = m_oldMarkerIds.size();

        lineListMarker.scale.x = m_lineWidth * borderWidth;

        geometry_msgs::Point point;

        // 4 vertical lines
        point.x = minCoords.x; point.y = minCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        point.x = minCoords.x; point.y = minCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);

        point.x = maxCoords.x; point.y = minCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        point.x = maxCoords.x; point.y = minCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);

        point.x = minCoords.x; point.y = maxCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        point.x = minCoords.x; point.y = maxCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);

        point.x = maxCoords.x; point.y = maxCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        point.x = maxCoords.x; point.y = maxCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);

        // 4 horizontal lines on bottom
        point.x = minCoords.x; point.y = minCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        point.x = maxCoords.x; point.y = minCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        
        point.x = maxCoords.x; point.y = minCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        point.x = maxCoords.x; point.y = maxCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        
        point.x = maxCoords.x; point.y = maxCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        point.x = minCoords.x; point.y = maxCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        
        point.x = minCoords.x; point.y = maxCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        point.x = minCoords.x; point.y = minCoords.y; point.z = minCoords.z; lineListMarker.points.push_back(point);
        
        // 4 horizontal lines on top
        point.x = minCoords.x; point.y = minCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);
        point.x = maxCoords.x; point.y = minCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);
        
        point.x = maxCoords.x; point.y = minCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);
        point.x = maxCoords.x; point.y = maxCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);
        
        point.x = maxCoords.x; point.y = maxCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);
        point.x = minCoords.x; point.y = maxCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);

        point.x = minCoords.x; point.y = maxCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);
        point.x = minCoords.x; point.y = minCoords.y; point.z = maxCoords.z; lineListMarker.points.push_back(point);

        m_markerArray.markers.push_back(lineListMarker);
        m_oldMarkerIds.push_back( std::make_pair<std::string, int>(lineListMarker.ns, lineListMarker.id) );
    }
}


void VolumeVisualizer::publish()
{
    m_markerArrayPublisher->publish(m_markerArray);
}


void VolumeVisualizer::clear()
{
    visualization_msgs::Marker deleteMarker;

    foreach(NsIdPair nsIdPair, m_oldMarkerIds) {
        deleteMarker.header.frame_id = m_markerFrame;
        deleteMarker.header.stamp = ros::Time::now();
        deleteMarker.ns = nsIdPair.first;
        deleteMarker.id = nsIdPair.second;
        deleteMarker.action = visualization_msgs::Marker::DELETE;
        m_markerArray.markers.push_back(deleteMarker);
    }
    
    m_oldMarkerIds.clear();
}


std_msgs::ColorRGBA VolumeVisualizer::getPaletteColor(int paletteColorIndex) const
{
    std_msgs::ColorRGBA color;
    
    if(m_colorScheme == "srl" || m_colorScheme == "srl_alternative")
    {
        // SRL People Tracker colors
        const size_t NUM_SRL_COLORS = 6, NUM_SRL_COLOR_SHADES = 4, NUM_SRL_TOTAL_COLORS = NUM_SRL_COLORS * NUM_SRL_COLOR_SHADES;
        const unsigned int spencer_colors[NUM_SRL_TOTAL_COLORS] = {
            0xC00000, 0xFF0000, 0xFF5050, 0xFFA0A0, // red
            0x00C000, 0x00FF00, 0x50FF50, 0xA0FFA0, // green
            0x0000C0, 0x0000FF, 0x5050FF, 0xA0A0FF, // blue
            0xF20A86, 0xFF00FF, 0xFF50FF, 0xFFA0FF, // magenta
            0x00C0C0, 0x00FFFF, 0x50FFFF, 0xA0FFFF, // cyan
            0xF5A316, 0xFFFF00, 0xFFFF50, 0xFFFFA0  // yellow
        };

        unsigned int rgb = 0;
        const unsigned int tableId = paletteColorIndex % NUM_SRL_TOTAL_COLORS;
        if(m_colorScheme == "srl") {
            // Colors in original order (first vary shade, then vary color)
            rgb = spencer_colors[tableId];
        }
        else if(m_colorScheme == "srl_alternative") {
            // Colors in alternative order (first vary color, then vary shade)
            unsigned int shadeIndex = tableId / NUM_SRL_COLORS;
            unsigned int colorIndex = tableId % NUM_SRL_COLORS;
            rgb = spencer_colors[colorIndex * NUM_SRL_COLOR_SHADES + shadeIndex];
        }

        color = colorFromInt(rgb);
    }
    else if(m_colorScheme == "rainbow" || m_colorScheme == "rainbow_bw")
    {
        const size_t NUM_COLOR = 10, NUM_BW = 4;
        const unsigned int rainbow_colors[NUM_COLOR + NUM_BW] = {
            0xaf1f90, 0x000846, 0x00468a, 0x00953d, 0xb2c908, 0xfcd22a, 0xffa800, 0xff4500, 0xe0000b, 0xb22222,
            0xffffff, 0xb8b8b8, 0x555555, 0x000000
        };

        color = colorFromInt(rainbow_colors[paletteColorIndex % (m_colorScheme == "rainbow" ? NUM_COLOR : (NUM_COLOR + NUM_BW))]);
    }
    else if(m_colorScheme == "flat")
    {
        const size_t NUM_COLOR = 10;
        const unsigned int flat_colors[NUM_COLOR] = {
            0x990033, 0xa477c4, 0x3498db, 0x1abc9c, 0x55e08f, 0xfff054, 0xef5523, 0xfe374a, 0xbaa891, 0xad5f43
        };

        color = colorFromInt(flat_colors[paletteColorIndex % NUM_COLOR]);
    }
    else if(m_colorScheme == "vintage")
    {
        const size_t NUM_COLOR = 10;
        const unsigned int vintage_colors[NUM_COLOR] = {
            0xd05e56, 0x797d88, 0x448d7a, 0xa5d1cd, 0x88a764, 0xebe18c, 0xd8a027, 0xffcc66, 0xdc3f1c, 0xff9966
        };

        color = colorFromInt(vintage_colors[paletteColorIndex % NUM_COLOR]);
    }
    else
    {
        // Constant color for all tracks
        color = colorFromInt(0xcccccc);
    }

    color.a = m_alpha;
    return color;
}


std_msgs::ColorRGBA VolumeVisualizer::colorFromInt(unsigned int rgb) const
{
    std_msgs::ColorRGBA color;
    color.r = ((rgb >> 16) & 0xff) / 255.0f,
    color.g = ((rgb >> 8)  & 0xff) / 255.0f,
    color.b = ((rgb >> 0)  & 0xff) / 255.0f;
    color.a = 1.0f;
    return color;
}