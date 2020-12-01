#include <memory>

#include<ros/ros.h>
#include<pcl/point_cloud.h>
#include<pcl/point_types.h>
#include<pcl_conversions/pcl_conversions.h>
#include<sensor_msgs/PointCloud2.h>
#include"Eigen/Dense"

#include<string>

//Object Detection(LiDAR)
#include "include/cnn_segmentation.h"

#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/pointcloudXYZIR.h>

// PCL
#include<sensor_msgs/PointCloud2.h>
#include <pcl/common/transforms.h>

pcl::PointCloud<pcl::PointXYZI> m_lidar_data;
std::string m_fixed_frame;

class CPerception
{
private:
    ros::NodeHandle nh;
    ros::Subscriber lidar_sub_;

    ros::Publisher clustering_pub;
    ros::Publisher boundingbox_pub;
    ros::Publisher speed_debug_pub;
    ros::Publisher obj_pub;
    CNNSegmentation seg_obj_detect;

public:
    ~CPerception()
    {
    }
    CPerception(ros::NodeHandle _nh)
    {
        nh=_nh;
        lidar_sub_ = nh.subscribe("/pandar", 1 ,&CPerception::Callback_lidar , this );

        boundingbox_pub = nh.advertise<visualization_msgs::MarkerArray>("/polygonbox_cube", 1);
        speed_debug_pub = nh.advertise<visualization_msgs::MarkerArray>("/ID", 1);
        clustering_pub = nh.advertise<sensor_msgs::PointCloud2> ("/lidar_clustering", 1);
        obj_pub = nh.advertise<polygon_msgs::polygonArray>("/obj_data", 1);
    }
    void Callback_lidar(sensor_msgs::PointCloud2Ptr scan)
    {
        pcl::PointCloud<velodyne_pointcloud::PointXYZIR> arr;
        m_fixed_frame = scan->header.frame_id;
        pcl::fromROSMsg(*scan, arr);
        m_lidar_data.clear();

        for(int i = 0; i < arr.size(); i++)
        {
            if(arr.at(i).ring != 39 && arr.at(i).ring != 0)
            {
                pcl::PointXYZI pt;
                pt._PointXYZI::x         = arr.at(i).x;
                pt._PointXYZI::y         = arr.at(i).y;
                pt._PointXYZI::z         = arr.at(i).z;
                pt._PointXYZI::intensity = arr.at(i).intensity;
                m_lidar_data.push_back(pt);
            }
        }

        lidar_thread();
    }
    void lidar_thread()
    {
        ///// seg Object Detection /////
        //Object Detection
        seg_obj_detect.CNN_seg(m_lidar_data, m_fixed_frame);

        // Object Det & Track Debug Publish
        speed_debug_pub.publish(seg_obj_detect.m_speed_result);
        boundingbox_pub.publish(seg_obj_detect.m_polygon_result);

        sensor_msgs::PointCloud2 seg_output;
        pcl::toROSMsg(seg_obj_detect.m_cluster_result, seg_output);
        sensor_msgs::PointCloud seg_output_arr;
        sensor_msgs::convertPointCloud2ToPointCloud(seg_output, seg_output_arr);
        seg_output.header.frame_id = "pos";
        clustering_pub.publish(seg_output);

        // Object Det & Track Result Publish
        obj_pub.publish(seg_obj_detect.m_polygon_pub);
        ////////////////////////////////////
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pollo_polygon_tracking_node");
    ros::NodeHandle nh;
    CPerception LOD(nh);

    ros::spin();
    return 0;
}

