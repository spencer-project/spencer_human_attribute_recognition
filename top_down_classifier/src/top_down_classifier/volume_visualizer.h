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