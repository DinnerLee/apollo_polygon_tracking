#ifndef CNN_SEGMENTATION_H
#define CNN_SEGMENTATION_H

#include <chrono>

#include <ros/ros.h>

#include <caffe/caffe.hpp>

// #include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/pointcloudXYZIR.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <rockauto_msgs/DetectedObjectArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <std_msgs/Header.h>

#include "cluster2d.h"
#include "feature_generator.h"
#include <include/TrackAssociation.h>

#include <Eigen/Dense>
#include <pcl/common/transformation_from_correspondences.h>

#include <polygon_msgs/polygon.h>
#include <polygon_msgs/polygonArray.h>
#include <polygon_msgs/polygon.h>
#include <pcl/common/transforms.h>
// #include "pcl_types.h"
// #include "modules/perception/obstacle/lidar/segmentation/cnnseg/cnn_segmentation.h"

#define __APP_NAME__ "lidar_cnn_seg_detect"

static int MAX_EXTRACT = 4;

struct minimum_pt{
    float dis;
    cv::Point3f pt;
};

struct CellStat
{
    CellStat() : point_num(0), valid_point_num(0)
    {
    }

    int point_num;
    int valid_point_num;
};

class CNNSegmentation
{
public:
    CNNSegmentation();
    void CNN_seg(pcl::PointCloud<pcl::PointXYZI> raw_point, std::string fixed_frame);
    polygon_msgs::polygonArray m_polygon_pub;

    //debug
    visualization_msgs::MarkerArray m_polygon_result;
    visualization_msgs::MarkerArray m_speed_result;
    pcl::PointCloud<pcl::PointXYZI> m_cluster_result;

private:
    ros::Subscriber sub_scan;
    bool init_num = false;
    std::vector<std::string> class_name;
    pcl::PointCloud <pcl::PointXYZI> m_lidar_data;
    std::string m_fixed_frame;
    double range_, score_threshold_;
    double max_distance_, min_distance_, clip_height_;
    int width_;
    int height_;
    bool use_constant_feature_;
    std_msgs::Header message_header_;
    std::string topic_src_;

    ros::Time run_time;

    // Object Tracking
    int min_num;
    TrackAssociation_pt Track;
    std::vector<cv::Point3f> m_local_track_pt;
    std::vector<cv::Point3f> m_global_track_pt;

    cv::Point3d m_max;
    cv::Point3d m_min;

    int gpu_device_id_;
    bool use_gpu_;

    std::shared_ptr<caffe::Net<float>> caffe_net_;

    // center offset prediction
    boost::shared_ptr<caffe::Blob<float>> instance_pt_blob_;
    // objectness prediction
    boost::shared_ptr<caffe::Blob<float>> category_pt_blob_;
    // fg probability prediction
    boost::shared_ptr<caffe::Blob<float>> confidence_pt_blob_;
    // object height prediction
    boost::shared_ptr<caffe::Blob<float>> height_pt_blob_;
    // raw features to be input into network
    boost::shared_ptr<caffe::Blob<float>> feature_blob_;
    // class prediction
    boost::shared_ptr<caffe::Blob<float>> class_pt_blob_;

    // clustering model for post-processing
    std::shared_ptr<Cluster2D> cluster2d_;

    // bird-view raw feature generator
    std::shared_ptr<FeatureGenerator> feature_generator_;

    bool init();

    std::vector<pcl::PointCloud<pcl::PointXYZI>> f_caffe_clustering(pcl::PointCloud<pcl::PointXYZI> points);
    void f_Tracking_make(std::vector<pcl::PointCloud<pcl::PointXYZI>> points);

    //Function Merge code
    pcl::PointCloud<pcl::PointXYZI> f_lidar_Passthrough( pcl::PointCloud<pcl::PointXYZI> point); //pcl ROI
    pcl::PointCloud<pcl::PointXYZI> f_roi_Passthrough( pcl::PointCloud<pcl::PointXYZI> point); //pcl ROI global

    bool segment(const pcl::PointCloud<pcl::PointXYZI> &pc_ptr,
                 const pcl::PointIndices &valid_idx,
                 rockauto_msgs::DetectedObjectArray &objects);

    std::vector<pcl::PointCloud<pcl::PointXYZI>> pubColoredPoints(const rockauto_msgs::DetectedObjectArray &objects);

    visualization_msgs::Marker f_make_polygon(std::vector<Point2f> hull, int num); // Make Object Polygon
    void f_make_ID(); //Make Object ID
    void f_make_total_polygon(); //Make Object ID
};

#endif //CNN_SEGMENTATION_H
