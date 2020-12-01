#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>

#define MAXN 1000
#define INF 1e9;
#define MAXLIFETIME 10
#define MAXCOST 255.0

static int TrackID = 0;
static double m_time;

using namespace cv;
using namespace std;

float euclideanDist(Point2f& p, Point2f& q);

class Track{
public:
    int ID;
    int life_time;
    std::string class_name;

    cv::Point3f local_cur_pos;
    std::vector<cv::Point3f> local_prev_pos;
    cv::Point3f local_prev_pos_;
    cv::Point3f local_predict_pos;

    std::vector<cv::Point2f> local_velocity_arr;
    std::vector<float> local_speed_arr;

    visualization_msgs::Marker polygon_data;

public:
    Track();
    ~Track(){}

    int get_life_time(){
        return life_time;
    }
    cv::Point3f get_local_prev_pos(){
        return local_prev_pos[local_prev_pos.size()-2];
    }
    cv::Point3f get_local_cur_pos(){
        return local_prev_pos[local_prev_pos.size()-1];
    }
    cv::Point2f get_local_velocity(){
        return local_velocity_arr[local_velocity_arr.size()-1];
    }
    float get_local_speed(){
        return local_speed_arr[local_speed_arr.size()-1];
    }

    void miss_Track();
    void miss_polygon_Track();
    void predict_Track();
    void speed_Track();
    void pos_mean_Track(Point3f local_obj);
    void tracking(Point3f local_obj, string name, visualization_msgs::Marker polygon);
};

class CHungarianAlgorithm_pt
{
public:
    int n, Match_num;                            // worker 수
    float label_x[MAXN], label_y[MAXN];           // label x, y
    int yMatch[MAXN];                           // y와 match되는 x
    int xMatch[MAXN];                           // x와 match되는 y
    bool S[MAXN], T[MAXN];                      // 알고리즘 상에 포함되는 vertex.
    float slack[MAXN];
    float slackx[MAXN];
    int parent[MAXN];                           // alternating path
    float cost[MAXN][MAXN];                       // cost
    float init_cost[MAXN][MAXN];                       // 초기 cost

    void init_labels();
    void update_labels();
    void add_to_tree(int x, int parent_x);
    void augment();
    void hungarian();


public:
    void HAssociation(vector<Track> &vc_tracks, vector<Point3f> &local_candi, float DIST_TH, vector<string> &name, visualization_msgs::MarkerArray &polygon);

    // Planning Number
    void Make_a_cost(vector<Track> &vc_tracks, vector<Point3f> &candi, bool check_max, float Dis_TH);

    CHungarianAlgorithm_pt();
    ~CHungarianAlgorithm_pt();
};



class TrackAssociation_pt
{
public:
    vector<Track> vc_groups;
public:
    TrackAssociation_pt(){}
    ~TrackAssociation_pt(){}

    void Track_pt(vector<Point3f> &local_candi, std::vector<string> class_name, visualization_msgs::MarkerArray polygon, ros::Duration run_time);
};


