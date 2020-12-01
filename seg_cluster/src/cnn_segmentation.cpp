#include "include/cnn_segmentation.h"

CNNSegmentation::CNNSegmentation()
{
    double distance = 80.0;

    //initial value
    m_max.x = distance; //ROI value
    m_max.y = distance;
    m_max.z = 1.0;

    m_min.x = (-1)*distance;
    m_min.y = (-1)*distance;
    m_min.z = -3.0;

    this->init();
}

bool CNNSegmentation::init()
{
    std::string proto_file;
    std::string weight_file;

    std::string path = "/home/a/catkin_ws/src/apollo_polygon_tracking/seg_cluster/";

    proto_file = path + "models/velodyne64/deploy.prototxt";
    weight_file = path + "models/velodyne64/deploy.caffemodel";

    range_ = 80.0;
    score_threshold_ = 0.4;
    width_ = 672;
    height_ = 672;

    min_distance_ = 3.0;
    max_distance_ = 80.0;
    clip_height_ = 5.0;

    use_constant_feature_ = false;
    use_gpu_ = true;
    gpu_device_id_ = 0;

    /// Instantiate Caffe net
    if (!use_gpu_)
    {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        std::cout << "using cpu" << std::endl;
    }
    else
    {
        caffe::Caffe::SetDevice(gpu_device_id_);
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::DeviceQuery();
        std::cout << "using gpu" << std::endl;
    }
    caffe_net_.reset(new caffe::Net<float>(proto_file, caffe::TEST));
    caffe_net_->CopyTrainedLayersFrom(weight_file);


    std::string instance_pt_blob_name = "instance_pt";
    instance_pt_blob_ = caffe_net_->blob_by_name(instance_pt_blob_name);
    CHECK(instance_pt_blob_ != nullptr) << "`" << instance_pt_blob_name
                                        << "` layer required";

    std::string category_pt_blob_name = "category_score";
    category_pt_blob_ = caffe_net_->blob_by_name(category_pt_blob_name);
    CHECK(category_pt_blob_ != nullptr) << "`" << category_pt_blob_name
                                        << "` layer required";

    std::string confidence_pt_blob_name = "confidence_score";
    confidence_pt_blob_ = caffe_net_->blob_by_name(confidence_pt_blob_name);
    CHECK(confidence_pt_blob_ != nullptr) << "`" << confidence_pt_blob_name
                                          << "` layer required";

    std::string height_pt_blob_name = "height_pt";
    height_pt_blob_ = caffe_net_->blob_by_name(height_pt_blob_name);
    CHECK(height_pt_blob_ != nullptr) << "`" << height_pt_blob_name
                                      << "` layer required";

    std::string feature_blob_name = "data";
    feature_blob_ = caffe_net_->blob_by_name(feature_blob_name);
    CHECK(feature_blob_ != nullptr) << "`" << feature_blob_name
                                    << "` layer required";

    std::string class_pt_blob_name = "class_score";
    class_pt_blob_ = caffe_net_->blob_by_name(class_pt_blob_name);
    CHECK(class_pt_blob_ != nullptr) << "`" << class_pt_blob_name
                                     << "` layer required";

    cluster2d_.reset(new Cluster2D());
    if (!cluster2d_->init(height_, width_, range_))
    {
        ROS_ERROR("[%s] Fail to Initialize cluster2d for CNNSegmentation", __APP_NAME__);
        return false;
    }

    feature_generator_.reset(new FeatureGenerator());
    if (!feature_generator_->init(feature_blob_.get(), use_constant_feature_, range_, width_, height_))
    {
        ROS_ERROR("[%s] Fail to Initialize feature generator for CNNSegmentation", __APP_NAME__);
        return false;
    }

    return true;
}

void CNNSegmentation::CNN_seg(pcl::PointCloud<pcl::PointXYZI> raw_point, std::string fixed_frame){
    // initialize
    m_fixed_frame = fixed_frame;

    m_polygon_pub.polygon.clear();

    m_polygon_result.markers.clear();
    m_speed_result.markers.clear();
    m_cluster_result.clear();

    m_global_track_pt.clear();
    m_local_track_pt.clear();
    ////////////////////////////

    // ROI Points extracted by HD map
    pcl::PointCloud<pcl::PointXYZI> l_passed_raw_point = f_lidar_Passthrough(raw_point);
    // main algorithm(segmentation & polygons making)
    ros::Time start1 = ros::Time::now();
    std::vector<pcl::PointCloud<pcl::PointXYZI>> l_cluster = f_caffe_clustering(l_passed_raw_point);
    ros::Time end1 = ros::Time::now();

    visualization_msgs::MarkerArray polygon_result;
    for(int i = 0; i < l_cluster.size(); i++){
        // Make Tracking Data
        std::vector<cv::Point2f> pt_arr;
        for(int j = 0; j < l_cluster.at(i).size(); j++){
            cv::Point2f pt;
            pt.x = l_cluster[i][j].x;
            pt.y = l_cluster[i][j].y;
            pt_arr.push_back(pt);
        }

        std::vector<cv::Point2f> hull(pt_arr.size());
        cv::convexHull(pt_arr,hull);

        polygon_result.markers.push_back(f_make_polygon(hull, i));
    }
    //////////////////////////////////////////////////////////////

    // Tracking
    f_Tracking_make(l_cluster);
    ros::Time start = ros::Time::now();
    ros::Duration time = start - run_time;

    Track.Track_pt(m_local_track_pt, class_name, polygon_result, time);

    run_time = ros::Time::now();

    //////////////////////////////////////////////////////////////

    // Object Detection & Tracking Result Data
    for(int i = 0; i < Track.vc_groups.size(); i++){
        if(Track.vc_groups.at(i).life_time >= MAX_EXTRACT){
            polygon_msgs::polygon arr;
            arr.class_name = Track.vc_groups[i].class_name;
            arr.velocity_x = Track.vc_groups[i].get_local_velocity().x;
            arr.velocity_y = Track.vc_groups[i].get_local_velocity().y;
            arr.speed = Track.vc_groups[i].get_local_speed();
            arr.local_x = Track.vc_groups[i].local_cur_pos.x;
            arr.local_y = Track.vc_groups[i].local_cur_pos.y;
            arr.markers = Track.vc_groups[i].polygon_data;

            m_polygon_pub.polygon.push_back(arr);
        }
    }
    ////////////////////////////////////////////
    for(int i = 0; i < l_cluster.size(); i++){
        m_cluster_result += l_cluster.at(i);
    }

    f_make_ID();
    f_make_total_polygon();
    std::cout << "running time : " << end1 - start1 << std::endl;
    //////////////////////////////////////////////////
}

// ROI-Filter
pcl::PointCloud<pcl::PointXYZI> CNNSegmentation::f_lidar_Passthrough(pcl::PointCloud<pcl::PointXYZI> point){ //장애물의 전체적인 범위 설정을 통해서 필요없는 부분 제거
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI> filter;
    pcl::PassThrough <pcl::PointXYZI> pass;

    *cloud = point;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(m_min.z, m_max.z);
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(m_min.x, m_max.x);
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(m_min.y, m_max.y);
    pass.filter(*cloud_filter);
    filter = *cloud_filter;

    return filter;
}

// Clustering - Main
std::vector<pcl::PointCloud<pcl::PointXYZI>> CNNSegmentation::f_caffe_clustering(pcl::PointCloud<pcl::PointXYZI> points){
    std::vector<pcl::PointCloud<pcl::PointXYZI>> result;

    pcl::PointIndices valid_idx;
    auto &indices = valid_idx.indices;
    indices.resize(points.size());
    std::iota(indices.begin(), indices.end(), 0);

    rockauto_msgs::DetectedObjectArray objects;
    segment(points, valid_idx, objects);

    result = pubColoredPoints(objects);

    return result;
}

// Clustering - segmentation
bool CNNSegmentation::segment(const pcl::PointCloud<pcl::PointXYZI> &pc_ptr,
                              const pcl::PointIndices &valid_idx,
                              rockauto_msgs::DetectedObjectArray &objects)
{
    int num_pts = static_cast<int>(pc_ptr.size());

    feature_generator_->generate(pc_ptr);

    // network forward process
    caffe_net_->Forward();
#ifndef USE_CAFFE_GPU
    //  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    //  int gpu_id = 0;
    //   caffe::Caffe::SetDevice(gpu_id);
    //    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    //    caffe::Caffe::DeviceQuery();
#endif

    // clutser points and construct segments/objects
    float objectness_thresh = 0.5;
    bool use_all_grids_for_clustering = true;
    cluster2d_->cluster(*category_pt_blob_, *instance_pt_blob_, pc_ptr,
                        valid_idx, objectness_thresh,
                        use_all_grids_for_clustering);
    cluster2d_->filter(*confidence_pt_blob_, *height_pt_blob_);
    cluster2d_->classify(*class_pt_blob_);
    float confidence_thresh = score_threshold_;
    float height_thresh = 0.2;
    int min_pts_num = 4;
    cluster2d_->getObjects(confidence_thresh, height_thresh, min_pts_num,
                           objects, message_header_, feature_generator_->height_map_);

    return true;
}

// Clustering - make_points
std::vector<pcl::PointCloud<pcl::PointXYZI>> CNNSegmentation::pubColoredPoints(const rockauto_msgs::DetectedObjectArray &objects_array)
{
    std::vector<pcl::PointCloud<pcl::PointXYZI>> cloud;

    for (size_t object_i = 0; object_i < objects_array.objects.size(); object_i++)
    {
        if(objects_array.objects[object_i].score < 0.1 && objects_array.objects[object_i].label == "unknwon")
            continue;
        if(objects_array.objects.at(object_i).label == "car" && objects_array.objects.at(object_i).score >= 0.4){
            class_name.push_back(objects_array.objects[object_i].label);
        }
        else{
            std::string arr = "unknwon";
            class_name.push_back(arr);
        }

        pcl::PointCloud<pcl::PointXYZI> pt;
        pcl::PointCloud<pcl::PointXYZI> object_cloud;
        pcl::fromROSMsg(objects_array.objects[object_i].pointcloud, object_cloud);

        pcl::PointXYZI ppt;

        for (size_t i = 0; i < object_cloud.size(); i++)
        {
            pcl::PointXYZI colored_point;
            colored_point.x = object_cloud[i].x;
            colored_point.y = object_cloud[i].y;
            colored_point.z = object_cloud[i].z;
            colored_point.intensity = object_i;
            pt.push_back(colored_point);

        }
        cloud.push_back(pt);
    }

    return cloud;
}

bool cmp(const minimum_pt & a, const minimum_pt & b)
{
    if (a.dis < b.dis) return true; // 제일 먼저 f를 기준으로 오름차순 정렬

    // 각 경우에 대하여 else를 고려할 필요가 없다.
    return false;
}

// Tracking - make_points
void CNNSegmentation::f_Tracking_make(std::vector<pcl::PointCloud<pcl::PointXYZI>> points){
    pcl::PointCloud<pcl::PointXYZI> pt_arr;

    for(int i = 0; i < points.size(); i++){
        pcl::PointXYZI pt_pcl;
        cv::Point3f pt_pt;
        std::vector<minimum_pt> feature_min;

        for(int j = 0; j < points.at(i).size(); j++){
            float dis = sqrt(pow(points[i][j].x, 2) + pow(points[i][j].y, 2));

            minimum_pt arr;
            arr.dis = dis;
            arr.pt.x = points[i][j].x;
            arr.pt.y = points[i][j].y;
            arr.pt.z = points[i][j].z;
            feature_min.push_back(arr);
        }
        sort(feature_min.begin(), feature_min.end(), cmp);

        float sum_x = 0.0;
        float sum_y = 0.0;

        int count = 10;
        if(feature_min.size() < 10)
            count = feature_min.size();

        for(int j = 0; j < count; j++){
            sum_x += feature_min.at(j).pt.x;
            sum_y += feature_min.at(j).pt.y;
        }

        float avg_x = sum_x/(float)count;
        float avg_y = sum_y/(float)count;

        pt_pcl.x = avg_x;
        pt_pcl.y = avg_y;
        pt_pcl.z = 0.05;
        pt_pcl.intensity = 255;

        pt_pt.x = avg_x;
        pt_pt.y = avg_y;
        pt_pt.z = 0.05;

        m_local_track_pt.push_back(pt_pt);
    }
}

////////// Debug Mode //////////
void CNNSegmentation::f_make_ID(){
    pcl::PointCloud<pcl::PointXYZ> pt_arr;

    for(int i = 0; i < Track.vc_groups.size(); i++){
        if(Track.vc_groups.at(i).life_time >= MAX_EXTRACT){
            string ID;

            ID = std::to_string(Track.vc_groups[i].ID);
            ID += "\n";
            ID += Track.vc_groups[i].class_name;
            ID += "\n";
            ID += "local x: ";
            ID += std::to_string(Track.vc_groups[i].get_local_velocity().x);
            ID.erase(ID.size()-4, ID.size());
            ID += "\n";
            ID += "local y: ";
            ID += std::to_string(Track.vc_groups[i].get_local_velocity().y);
            ID.erase(ID.size()-4, ID.size());
            ID += "\n";
            ID += "local: ";
            ID += std::to_string(Track.vc_groups[i].get_local_speed());
            ID.erase(ID.size()-4, ID.size());

            visualization_msgs::Marker marker;
            marker.header.frame_id = "pos";
            marker.header.stamp = ros::Time::now();
            marker.ns = "ID";
            marker.id = i;
            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = Track.vc_groups.at(i).local_cur_pos.x;
            marker.pose.position.y = Track.vc_groups.at(i).local_cur_pos.y;
            marker.pose.position.z = 1.0;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.text = ID;

            marker.scale.x = 1.0;
            marker.scale.y = 1.0;
            marker.scale.z = 1.0;

            marker.color.r = 1.0f;
            marker.color.g = 1.0f;
            marker.color.b = 1.0f;
            marker.color.a = 1.0;

            marker.lifetime = ros::Duration(0.1);
            m_speed_result.markers.push_back(marker);
        }
    }
}

void CNNSegmentation::f_make_total_polygon(){
    pcl::PointCloud<pcl::PointXYZ> pt_arr;

    for(int i = 0; i < Track.vc_groups.size(); i++){
        if(Track.vc_groups.at(i).life_time >= MAX_EXTRACT){
            Track.vc_groups[i].polygon_data.header.stamp = ros::Time::now();
            Track.vc_groups[i].polygon_data.header.frame_id = "pos";
            Track.vc_groups[i].polygon_data.ns = "adaptive_clustering";
            Track.vc_groups[i].polygon_data.id = i;

            Track.vc_groups[i].polygon_data.lifetime = ros::Duration(0.1);
            m_polygon_result.markers.push_back(Track.vc_groups[i].polygon_data);
        }
    }
}

visualization_msgs::Marker CNNSegmentation::f_make_polygon(std::vector<cv::Point2f> hull, int num){
    pcl::PointCloud<pcl::PointXYZ> pt_arr;

    for(int i = 0; i < hull.size(); i++){
        pcl::PointXYZ pt;
        pt.x = hull[i].x;
        pt.y = hull[i].y;
        pt.z = 0.05;

        pt_arr.push_back(pt);
    }

    //Drawing Rviz
    visualization_msgs::Marker marker;
    marker.header.stamp = ros::Time::now();
    marker.header.frame_id = "pos";
    marker.ns = "polygon";
    marker.id = num;
    marker.type = visualization_msgs::Marker::LINE_STRIP;

    geometry_msgs::Point p[pt_arr.size()+1];

    for(int i = 0; i < pt_arr.size(); i++){
        p[i].x = pt_arr.at(i).x;
        p[i].y = pt_arr.at(i).y;
        p[i].z = 0.05;
    }
    p[pt_arr.size()].x = pt_arr.at(0).x;
    p[pt_arr.size()].y = pt_arr.at(0).y;
    p[pt_arr.size()].z = 0.05;

    for(int k =  0; k < pt_arr.size(); k++){
        marker.points.push_back(p[k]);
    }

    marker.scale.x = 0.05;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;

    return marker;
}
