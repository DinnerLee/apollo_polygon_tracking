#include "include/TrackAssociation.h"

float euclideanDist(Point3f &p, Point3f &q) {
    Point2f a;
    a.x= p.x;
    a.y = p.y;

    Point2f b;
    b.x = q.x;
    b.y = q.y;

    Point2f diff = a - b;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

// Track Data
Track::Track()
{
    life_time = -1;
    class_name = "unkwon";
}
void Track::miss_Track(){
    if(life_time > -1){
        life_time--;
        miss_polygon_Track();

        // Predict Point update
        local_cur_pos = local_predict_pos;
        local_prev_pos.erase(local_prev_pos.begin());
        local_prev_pos.push_back(local_cur_pos);

        // Speed & Predict Point update
        speed_Track();
        predict_Track();
    }

    if (life_time == -1)
        return;
}
void Track::miss_polygon_Track(){
    for(int i = 0; i < polygon_data.points.size(); i++){
        polygon_data.points[i].x += local_predict_pos.x - local_cur_pos.x;
        polygon_data.points[i].y += local_predict_pos.y - local_cur_pos.y;
    }
}
void Track::predict_Track(){
    // Predict Tracking Point
    if(local_velocity_arr.size() > 0){
        cv::Point3f pt;
        pt.x = local_cur_pos.x + get_local_velocity().x*m_time;
        pt.y = local_cur_pos.y + get_local_velocity().y*m_time;
        pt.z = local_cur_pos.z;

        local_predict_pos = pt;
    }
    else{
        local_predict_pos = local_cur_pos;
    }

    if(local_velocity_arr.size() > 0){
        cv::Point3f pt;
        pt.x = local_cur_pos.x + get_local_velocity().x*m_time;
        pt.y = local_cur_pos.y + get_local_velocity().y*m_time;
        pt.z = local_cur_pos.z;

        local_predict_pos = pt;
    }
    else{
        local_predict_pos = local_cur_pos;
    }
}
void Track::speed_Track(){
    // Velocity & Speed Tracking
    if(life_time >= 2){
        // Mean Filter
        cv::Point2f local_velocity_pt;
        cv::Point2f global_velocity_pt;
        cv::Point2f global_speed;
        double speed_x;
        double speed_y;
        double speed;

        local_velocity_pt.x = (local_cur_pos.x - get_local_prev_pos().x)/m_time;
        local_velocity_pt.y = (local_cur_pos.y - get_local_prev_pos().y)/m_time;

        speed_x = pow(local_cur_pos.x - get_local_prev_pos().x, 2);
        speed_y = pow(local_cur_pos.y - get_local_prev_pos().y, 2);
        speed = sqrt(speed_x + speed_y)/m_time;

        // local x,y-axix velocity
        if(local_velocity_arr.size() < 3){
            local_velocity_arr.push_back(local_velocity_pt);
        }
        else
        {
            local_velocity_arr.erase(local_velocity_arr.begin());
            local_velocity_arr.push_back(local_velocity_pt);

            local_velocity_pt.x = 0.0;
            local_velocity_pt.y = 0.0;

            double count = 0.0;
            for(int j = 0; j < local_velocity_arr.size(); j++){
                local_velocity_pt.x += (j+1)*local_velocity_arr[j].x;
                local_velocity_pt.y += (j+1)*local_velocity_arr[j].y;
                count += (j+1);
            }
            local_velocity_pt.x /= count;
            local_velocity_pt.y /= count;

            local_velocity_arr[local_velocity_arr.size()-1].x = local_velocity_pt.x;
            local_velocity_arr[local_velocity_arr.size()-1].y = local_velocity_pt.y;
        }

        // Global speed
        if(local_speed_arr.size() < 3){
            local_speed_arr.push_back(speed);
        }
        else
        {
            local_speed_arr.erase(local_speed_arr.begin());
            local_speed_arr.push_back(speed);

            speed = 0.0;

            double count = 0.0;
            for(int j = 0; j < local_speed_arr.size(); j++){
                speed += (j+1)*local_speed_arr[j];
                count += (j+1);
            }
            speed /= count;

            local_speed_arr[local_speed_arr.size()-1] = speed;
        }
    }
    else{
        // No prev Data
        cv::Point2f local_velocity_pt;
        local_velocity_pt.x = 0.0;
        local_velocity_pt.y = 0.0;
        cv::Point2f global_velocity_pt;
        global_velocity_pt.x = 0.0;
        global_velocity_pt.y = 0.0;
        double speed = 0.0;

        local_velocity_arr.push_back(global_velocity_pt);
        local_speed_arr.push_back(speed);
    }
}
void Track::pos_mean_Track(Point3f local_obj){
    // local Position
    if(local_prev_pos.size() < 3){
        local_prev_pos.push_back(local_obj);
        local_cur_pos = local_obj;
    }
    else{
        local_prev_pos.erase(local_prev_pos.begin());
        local_prev_pos.push_back(local_obj);

        cv::Point3f arr;
        float count = 0.0;
        for(int i = 0; i < local_prev_pos.size(); i++){
            arr.x += (i+1)*local_prev_pos[i].x;
            arr.y += (i+1)*local_prev_pos[i].y;
            arr.z = local_prev_pos[i].z;

            count += (i+1);
        }

        arr.x /= count;
        arr.y /= count;

        local_prev_pos[local_prev_pos.size()-1] = arr;
        local_cur_pos = arr;
    }
}

void Track::tracking(Point3f local_obj, std::string name, visualization_msgs::Marker polygon)
{
    pos_mean_Track(local_obj);
    class_name = name;
    polygon_data = polygon;

    // Life_time Checking
    if (life_time == -1)
    {
        ID = TrackID++;
        life_time = 1;
    }
    else
    {
        if (life_time < MAXLIFETIME)
        {
            life_time++;
        }
    }

    speed_Track();
    predict_Track();
}

// HungarianAlgorithm
CHungarianAlgorithm_pt::CHungarianAlgorithm_pt()
{
}
CHungarianAlgorithm_pt::~CHungarianAlgorithm_pt()
{
}

void CHungarianAlgorithm_pt::Make_a_cost(vector<Track> &vc_tracks, vector<Point3f> &candi, bool check_max, float Dis_TH){
    float xmax = 0.0;
    if(check_max == true){
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j >= candi.size())
                {
                    cost[i][j] = 255;
                    init_cost[i][j] = 255;
                    xmax = max(xmax, cost[i][j]);
                }
                else
                {
                    Point3f predict_candi_pt = vc_tracks[i].local_predict_pos;
                    Point3f candi_pt = candi[j];

                    float Dist = euclideanDist(predict_candi_pt,candi_pt);
                    if(Dist >= Dis_TH){
                        cost[i][j] = 255;
                        init_cost[i][j] = 255;
                        xmax = max(xmax, cost[i][j]);
                    }
                    else{
                        cost[i][j] = Dist;
                        init_cost[i][j] = Dist;
                        xmax = max(xmax, cost[i][j]);
                    }

                }

            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cost[i][j] = xmax - cost[i][j];
            }
        }
    }
    else{
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i >= vc_tracks.size())
                {
                    cost[i][j] = 255;
                    init_cost[i][j] = 255;
                    xmax = max(xmax, cost[i][j]);
                }
                else
                {
                    Point3f predict_candi_pt = vc_tracks[i].local_predict_pos;
                    Point3f candi_pt = candi[j];
                    float Dist = euclideanDist(predict_candi_pt,candi_pt);
                    if(Dist >= Dis_TH){
                        cost[i][j] = 255;
                        init_cost[i][j] = 255;
                        xmax = max(xmax, cost[i][j]);
                    }
                    else{
                        cost[i][j] = Dist;
                        init_cost[i][j] = Dist;
                        xmax = max(xmax, cost[i][j]);
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cost[i][j] = xmax - cost[i][j];
            }
        }
    }
}

void CHungarianAlgorithm_pt::HAssociation(vector<Track> &vc_tracks, vector<Point3f> &local_candi, float DIST_TH, vector<string> &name, visualization_msgs::MarkerArray &polygon)
{
    if (vc_tracks.size() > local_candi.size()) // 기존 트랙의 수가 더많을때
    {
        n = vc_tracks.size();

        Make_a_cost(vc_tracks, local_candi, true, DIST_TH);

        hungarian();
        for (int x = 0; x < n; x++)
        {
            if (init_cost[x][xMatch[x]] > DIST_TH) //  이하이면
            {
                vc_tracks[x].miss_Track();
                if (xMatch[x] < local_candi.size()) // 둘다 유효할떄
                {
                    Track newTrack;
                    newTrack.tracking(local_candi[xMatch[x]], name[xMatch[x]], polygon.markers.at(xMatch[x]));
                    vc_tracks.push_back(newTrack);
                }
            }
            else
            {
                vc_tracks[x].tracking(local_candi[xMatch[x]], name[xMatch[x]], polygon.markers.at(xMatch[x]));
            }
        }
    }
    else // 기존 트랙수가 더적을때
    {
        n = local_candi.size();

        Make_a_cost(vc_tracks, local_candi, false, DIST_TH);

        hungarian();

        for (int y = 0; y < n; y++)
        {
            if (init_cost[yMatch[y]][y] > DIST_TH)
            {
                if (yMatch[y] < vc_tracks.size()) // 둘다 유효할때
                {
                    vc_tracks[yMatch[y]].miss_Track();
                }
                Track newTrack;
                newTrack.tracking(local_candi[y], name[y], polygon.markers.at(y));
                vc_tracks.push_back(newTrack);
            }
            else
            {
                vc_tracks[yMatch[y]].tracking(local_candi[y], name[y], polygon.markers.at(y));
            }
        }
    }
}

void CHungarianAlgorithm_pt::init_labels()
{
    memset(label_x, 0, sizeof(label_x));
    memset(label_y, 0, sizeof(label_y));      // y label은 모두 0으로 초기화.

    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++)
            label_x[x] = max(label_x[x], cost[x][y]);    // cost중에 가장 큰 값을 label 값으로 잡음.
}

void CHungarianAlgorithm_pt::update_labels()
{
    float delta = (float)INF;

    // slack통해서 delta값 계산함.
    for (int y = 0; y < n; y++)
        if (!T[y]) delta = min(delta, slack[y]);

    for (int x = 0; x < n; x++)
        if (S[x]) label_x[x] -= delta;
    for (int y = 0; y < n; y++) {
        if (T[y]) label_y[y] += delta;
        else slack[y] -= delta;
    }
}

void CHungarianAlgorithm_pt::add_to_tree(int x, int parent_x)
{
    S[x] = true;            // S집합에 포함.
    parent[x] = parent_x;   // augmenting 할때 필요.

    for (int y = 0; y < n; y++) {                                   // 새 노드를 넣었으니, slack 갱신해야함.
        if (label_x[x] + label_y[y] - cost[x][y] < slack[y]) {
            slack[y] = label_x[x] + label_y[y] - cost[x][y];
            slackx[y] = x;
        }
    }
}

void CHungarianAlgorithm_pt::augment()
{
    if (Match_num == n) return;
    int root;   // 시작지점.
    queue<int> q;

    memset(S, false, sizeof(S));
    memset(T, false, sizeof(T));
    memset(parent, -1, sizeof(parent));

    // root를 찾음. 아직 매치안된 y값을 찾음ㅇㅇ.
    for (int x = 0; x < n; x++) {
        if (xMatch[x] == -1) {
            q.push(root = x);
            parent[x] = -2;
            S[x] = true;
            break;
        }
    }

    // slack 초기화.
    for (int y = 0; y < n; y++) {
        slack[y] = label_x[root] + label_y[y] - cost[root][y];
        slackx[y] = root;
    }

    int x, y;
    // augment function
    while (1) {
        // bfs cycle로 tree building.
        while (!q.empty()) {
            x = q.front(); q.pop();
            for (y = 0; y < n; y++) {
                if (cost[x][y] == label_x[x] + label_y[y] && !T[y]) {
                    if (yMatch[y] == -1) break;
                    T[y] = true;
                    q.push(yMatch[y]);
                    add_to_tree(yMatch[y], x);
                }
            }
            if (y < n) break;
        }
        if (y < n) break;

        while (!q.empty()) q.pop();

        update_labels(); // 증가경로가 없다면 label 향상ㄱ.

        // label 향상을 통해서 equality graph의 새 edge를 추가함.
        // !T[y] && slack[y]==0 인 경우에만 add 할 수 있음.
        for (y = 0; y < n; y++) {
            if (!T[y] && slack[y] == 0) {
                if (yMatch[y] == -1) {          // 증가경로 존재.
                    x = slackx[y];
                    break;
                }
                else {
                    T[y] = true;
                    if (!S[yMatch[y]]) {
                        q.push(yMatch[y]);
                        add_to_tree(yMatch[y], slackx[y]);
                    }
                }
            }
        }
        if (y < n) break;  // augment path found;
    }

    if (y < n) {        // augment path exist
        Match_num++;

        for (int cx = x, cy = y, ty; cx != -2; cx = parent[cx], cy = ty) {
            ty = xMatch[cx];
            yMatch[cy] = cx;
            xMatch[cx] = cy;
        }
        augment();  // 새 augment path 찾음.
    }
}

void CHungarianAlgorithm_pt::hungarian()
{
    Match_num = 0;

    memset(xMatch, -1, sizeof(xMatch));
    memset(yMatch, -1, sizeof(yMatch));

    init_labels();
    augment();
}
////////////////////////////////////////////////////////////////////////////////////////

void TrackAssociation_pt::Track_pt(vector<Point3f> &local_candi, std::vector<std::string> class_name, visualization_msgs::MarkerArray polygon, ros::Duration run_time) // 현재 측정치
{
    m_time = run_time.toSec();
    if (vc_groups.size() == 0) // 기존 트랙이 없는경우
    {
        // insert Track Data
        for (int i = 0; i < local_candi.size(); i++)
        {
            Track newTrack;
            newTrack.tracking(local_candi[i], class_name[i], polygon.markers[i]);
            vc_groups.push_back(newTrack);
        }
        //////////////////////////////////////////////////////////////
    }
    else // 기존 트랙이 있는경우
    {
        float L_minCost = 4.0;

        if (local_candi.size() == 1 && vc_groups.size() == 1)
        {
            cv::Point3f candi0_pt = local_candi[0];
            cv::Point3f prev0_pt = vc_groups[0].get_local_prev_pos();
            cv::Point3f predict0_pt = vc_groups[0].local_predict_pos;

            float cost = euclideanDist(candi0_pt, prev0_pt);

            if (cost > L_minCost)
            {
                Track newTrack;
                newTrack.tracking(local_candi[0], class_name[0], polygon.markers[0]);
                vc_groups.push_back(newTrack);
                vc_groups[0].miss_Track();
                if (vc_groups[0].life_time == -1)
                    vc_groups.erase(vc_groups.begin());
            }
            else
            {
                vc_groups[0].tracking(local_candi[0], class_name[0], polygon.markers[0]);
            }

        }
        else
        {
            CHungarianAlgorithm_pt h;
            h.HAssociation(vc_groups, local_candi, L_minCost, class_name, polygon);

            for (int i = 0; i < vc_groups.size(); i++)
            {
                if (vc_groups[i].life_time == -1)
                {
                    vc_groups.erase(vc_groups.begin() + i);
                    i--;
                }
            }
        }
    }
}
