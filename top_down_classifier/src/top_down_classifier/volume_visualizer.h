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

#ifndef VOLUME_VISUALIZER_H_
#define VOLUME_VISUALIZER_H_

#include "volume.h"

#include <list>
#include <utility>
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>


/// Visualizes volumes using Rviz Marker messages.
class VolumeVisualizer {
public:
    VolumeVisualizer();

    /// Visualizes the volume as an Rviz marker.
    void visualize(const Volume& volume, int paletteColorIndex = 0, const std::string& ns = "Volumes", unsigned int borderColor = 0x000000, float borderWidth = 1.0f);

    // Removes all existing visualized volumes.
    void clear();

    // Publishes all changes.
    void publish();

private:
    typedef std::pair<std::string, int> NsIdPair;

    std_msgs::ColorRGBA getPaletteColor(int paletteColorIndex) const;
    std_msgs::ColorRGBA colorFromInt(unsigned int rgb) const;

    std::list<NsIdPair > m_oldMarkerIds;
    boost::shared_ptr<ros::Publisher> m_markerArrayPublisher;

    visualization_msgs::MarkerArray m_markerArray;
    std::string m_markerFrame, m_colorScheme;
    double m_alpha, m_lineWidth;
    bool m_showWireframe, m_showSolid;
};


#endif // VOLUME_VISUALIZER_H_