#include "tessellation_generator.h"

#include <ros/ros.h>
#include <map>
#include <cmath>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

TessellationGenerator::TessellationGenerator(const Volume& parentVolume, std::vector<pcl::PointXYZ> voxelAspectRatios, std::vector<float> voxelSizeIncrements, float minVoxelSize, bool overlapEnabled)
    : m_parentVolume(parentVolume)
{
    m_minVoxelSize = minVoxelSize;
    m_maxBorderWithoutVoxel = 0.25;
    m_parentVolumeSize = m_parentVolume.getSize();

    m_voxelAspectRatios = voxelAspectRatios;
    m_voxelSizeIncrements = voxelSizeIncrements;
    m_overlapEnabled = overlapEnabled;
}


Tessellation TessellationGenerator::generateTessellation(const pcl::PointXYZ& voxelSize)
{    
    std::vector<Volume> volumes;

    // safe it is for sure round, avoiud rounding errs  <-- wtf?
    double numrep_w = ceil(m_parentVolumeSize.y / voxelSize.y);
    double numrep_d = ceil(m_parentVolumeSize.x / voxelSize.x);
    double numrep_h = ceil(m_parentVolumeSize.z / voxelSize.z);
        
    for(unsigned int w = 0; w < (unsigned int) numrep_w; w++)
    {
        for(unsigned int d = 0; d < (unsigned int) numrep_d; d++)
        {
            for(unsigned int h = 0; h < (unsigned int) numrep_h; h++)
            {
                pcl::PointXYZ minCoords, maxCoords;
                minCoords.x = w * voxelSize.y;
                minCoords.y = d * voxelSize.x;
                minCoords.z = h * voxelSize.z;

                maxCoords.x = (w+1) * voxelSize.y;
                maxCoords.y = (d+1) * voxelSize.x;
                maxCoords.z = (h+1) * voxelSize.z;
                
                volumes.push_back( Volume(minCoords, maxCoords) );   
                
                // assert
                if(maxCoords.x > m_parentVolumeSize.x + FLT_EPSILON  || maxCoords.y > m_parentVolumeSize.y + FLT_EPSILON ||  maxCoords.z > m_parentVolumeSize.z + FLT_EPSILON)
                {
                    //ROS_ERROR("FATAL ERROR: COORDS OVER THE BOUNDARIES: %g %g %g\n", maxCoords.x, maxCoords.y, maxCoords.z);
                    //exit(1);
                } 
            }
        }
    }

    centerVolumes(volumes);
    return Tessellation(voxelSize, volumes, false);
}


Tessellation TessellationGenerator::generateOverlappingTessellation(const pcl::PointXYZ& voxelSize)
{    
    std::vector<Volume> volumes;

    // safe it is for sure round, avoiud rounding errs  <-- wtf?
    double numrep_w = ceil(m_parentVolumeSize.y / voxelSize.y);
    double numrep_d = ceil(m_parentVolumeSize.x / voxelSize.x);
    double numrep_h = ceil(m_parentVolumeSize.z / voxelSize.z);
   
    // overlapping tesselation
    double offsw = 0, offsd = 0, offsh = 0;
    if((unsigned int) numrep_w > 1) offsw = voxelSize.y / 2.0;
    if((unsigned int) numrep_d > 1) offsd = voxelSize.x / 2.0;        
    if((unsigned int) numrep_h > 1) offsh = voxelSize.z / 2.0;

    for(int w = 0; w < (int) numrep_w; w++)
    {
        for(int d = 0; d < (int) numrep_d; d++)
        {
            for(int h = 0; h < (int) numrep_h; h++)
            {
                pcl::PointXYZ minCoords, maxCoords;
                minCoords.x = d * voxelSize.x + offsd;
                minCoords.y = w * voxelSize.y + offsw;
                minCoords.z = h * voxelSize.z + offsh;

                maxCoords.x = (d+1) * voxelSize.x + offsd;
                maxCoords.y = (w+1) * voxelSize.y + offsw;
                maxCoords.z = (h+1) * voxelSize.z + offsh;
                
                // assert
                if(maxCoords.y > m_parentVolumeSize.y + m_maxBorderWithoutVoxel  || maxCoords.x > m_parentVolumeSize.x + m_maxBorderWithoutVoxel || maxCoords.z > m_parentVolumeSize.z + m_maxBorderWithoutVoxel
                || minCoords.y < -m_maxBorderWithoutVoxel || minCoords.x < -m_maxBorderWithoutVoxel || minCoords.z < -m_maxBorderWithoutVoxel)
                {
                     break;
                }
                else
                {
                    volumes.push_back( Volume(minCoords, maxCoords) );   
                }
            }
        }
    }

    centerVolumes(volumes);
    return Tessellation(voxelSize, volumes, true);
}


void TessellationGenerator::centerVolumes(std::vector<Volume>& volumes)
{
    Volume boundingVolume = Volume::getBoundingVolume(volumes);
    pcl::PointXYZ boundingVolumeSize = boundingVolume.getSize();

    foreach(Volume& volume, volumes)
    {
        volume.getMinCoordinates().x -= boundingVolumeSize.x / 2.0;
        volume.getMinCoordinates().y -= boundingVolumeSize.y / 2.0;
        volume.getMaxCoordinates().x -= boundingVolumeSize.x / 2.0;
        volume.getMaxCoordinates().y -= boundingVolumeSize.y / 2.0;
    }
}


void TessellationGenerator::generateTessellations(std::list<Tessellation>& tessellations)
{
    double curw, curd, curh;
    std::map<int, bool> hashTable;
    
    foreach(float voxelSizeIncrement, m_voxelSizeIncrements)
    {
        foreach(pcl::PointXYZ voxelAspectRatio, m_voxelAspectRatios)
        {
            curw = voxelSizeIncrement * voxelAspectRatio.y;
            curd = voxelSizeIncrement * voxelAspectRatio.x;            
            curh = voxelSizeIncrement * voxelAspectRatio.z;
            
            pcl::PointXYZ voxelSize;
            voxelSize.y = curw; voxelSize.x = curd; voxelSize.z = curh;

            bool isok = true;
            int w = 1;

            while(isok)
            {
                voxelSize.y = curw * w;
                voxelSize.x = curd * w;
                voxelSize.z = curh * w;
                
                if(voxelSize.y > m_parentVolumeSize.y ||  voxelSize.z  > m_parentVolumeSize.z  || voxelSize.x > m_parentVolumeSize.x || voxelSize.y < m_minVoxelSize ||  voxelSize.z  < m_minVoxelSize  || voxelSize.x < m_minVoxelSize)
                {
                    //ROS_WARN_STREAM("Voxel size not valid: " << voxelSize.x << ", " << voxelSize.y << ", " << voxelSize.z);
                    isok = false; 
                }
                else
                {
                    //ROS_INFO_STREAM("Voxel size valid: " << voxelSize.x << ", " << voxelSize.y << ", " << voxelSize.z);
                    
                    double qw = ceil(m_parentVolumeSize.y / voxelSize.y);
                    double rw = m_parentVolumeSize.y - qw * voxelSize.y;
                    double qd = ceil(m_parentVolumeSize.x / voxelSize.x);
                    double rd = m_parentVolumeSize.x - qd * voxelSize.x;
                    double qh = ceil(m_parentVolumeSize.z / voxelSize.z);
                    double rh = m_parentVolumeSize.z - qh * voxelSize.z;

                    // FIXME: Test
                    rw = std::abs(rw); rd = std::abs(rd); rh = std::abs(rh);
                    
                    if(rw < m_maxBorderWithoutVoxel && rd < m_maxBorderWithoutVoxel && rh < m_maxBorderWithoutVoxel)
                    {
                        ROS_INFO_STREAM("Got valid tessellation for voxel size: "  << voxelSize.x << ", " << voxelSize.y << ", " << voxelSize.z);
                        ROS_INFO("rd, rw, rh: %.3g %.3g %.3g", rd, rw, rh);

                        // assumes smaller dim < 1e-5
                        int hash_idx = voxelSize.y*10000  + voxelSize.x*100000 + voxelSize.z*1000000;
                        if(!hashTable[hash_idx])
                        {
                            Tessellation tessellation = generateTessellation(voxelSize);
                            tessellations.push_back(tessellation);

                            if(m_overlapEnabled) {
                                Tessellation overlappingTessellation = generateOverlappingTessellation(voxelSize);
                                tessellations.push_back(overlappingTessellation);
                            }

                            hashTable[hash_idx] = true;
                        }                                
                    }
                    else {
                        ROS_ERROR("Discarding tessellation, voxels of following size would not sufficiently cover parent volume: %.3g %.3g %.3g -- rd, rw, rh: %.3g %.3g %.3g", voxelSize.x, voxelSize.y, voxelSize.z, rd, rw, rh);
                    }
                }
                w++;
            }   
        }
    }       
}   
