#include <include/feature_generator.h>

bool FeatureGenerator::init(caffe::Blob<float>* out_blob, bool use_constant_feature, int range, int width, int height)
{
    out_blob_ = out_blob;

    // raw feature parameters
    // range_ = 60;
    // width_ = 512;
    // height_ = 512;

    range_ = range;
    width_ = width;
    height_ = height;
    min_height_ = -5.0;
    max_height_ = 5.0;
    CHECK_EQ(width_, height_)
            << "Current implementation version requires input_width == input_height.";

    // set output blob and log lookup table
    if(use_constant_feature){
        out_blob_->Reshape(1, 8, height_, width_);
    }else{
        out_blob_->Reshape(1, 6, height_, width_);
    }
    log_table_.resize(256);
    for (size_t i = 0; i < log_table_.size(); ++i) {
        log_table_[i] = std::log1p(static_cast<float>(i));
    }

    float* out_blob_data = nullptr;
    out_blob_data = out_blob_->mutable_cpu_data();
    // the pretrained model inside apollo project don't use the constant feature like direction_data_ and distance_data_
    int channel_index = 0;
    max_height_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
    mean_height_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
    count_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
    if(use_constant_feature){
        direction_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
    }
    top_intensity_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
    mean_intensity_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
    if(use_constant_feature){
        distance_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
    }
    nonempty_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
    CHECK_EQ(out_blob_->offset(0, channel_index), out_blob_->count());
    if(use_constant_feature){
        // compute direction and distance features
        int siz = height_ * width_;
        std::vector<float> direction_data(siz);
        std::vector<float> distance_data(siz);

        for (int row = 0; row < height_; ++row) {
            for (int col = 0; col < width_; ++col) {
                int idx = row * width_ + col;
                // * row <-> x, column <-> y
                float center_x = Pixel2Pc(row, height_, range_);
                float center_y = Pixel2Pc(col, width_, range_);
                constexpr double K_CV_PI = 3.1415926535897932384626433832795;
                direction_data[idx] =
                        static_cast<float>(std::atan2(center_y, center_x) / (2.0 * K_CV_PI));
                distance_data[idx] =
                        static_cast<float>(std::hypot(center_x, center_y) / 60.0 - 0.5);
            }
        }
        caffe::caffe_copy(siz, direction_data.data(), direction_data_);
        caffe::caffe_copy(siz, distance_data.data(), distance_data_);
    }
    return true;
}

float FeatureGenerator::logCount(int count)
{
    if (count < static_cast<int>(log_table_.size())) {
        return log_table_[count];
    }
    return std::log(static_cast<float>(1 + count));
}

float middle_point(std::vector<float> data){
    float result = 0.0;

    std::sort(data.begin(), data.end());
    if(data.size() != 0){
        result = data.at(data.size()/2);
    }

    return result;
}

void FeatureGenerator::generate(
        const pcl::PointCloud<pcl::PointXYZI>& pc_ptr) {
    const auto& points = pc_ptr;

    // DO NOT remove this line!!!
    // Otherwise, the gpu_data will not be updated for the later frames.
    // It marks the head at cpu for blob.
    out_blob_->mutable_cpu_data();

    int siz = height_ * width_;
    height_map_.resize(siz);

    // height threshold 0.05
    double m_height_diff_threshold = 0.05;

    std::vector<float> min;
    std::vector<float> max;
    std::vector<unsigned int> num;
    std::vector<bool> init;

    std::vector<std::vector<float>> total_points;

    min.assign(siz, 5);
    max.assign(siz, -5);
    num.assign(siz, 0);
    init.assign(siz, false);

    total_points.resize(siz);

    caffe::caffe_set(siz, float(-5), max_height_data_);
    caffe::caffe_set(siz, float(0), mean_height_data_);
    caffe::caffe_set(siz, float(0), count_data_);
    caffe::caffe_set(siz, float(0), top_intensity_data_);
    caffe::caffe_set(siz, float(0), mean_intensity_data_);
    caffe::caffe_set(siz, float(0), nonempty_data_);

    map_idx_.resize(points.size());
    float inv_res_x =
            0.5 * static_cast<float>(width_) / static_cast<float>(range_);
    float inv_res_y =
            0.5 * static_cast<float>(height_) / static_cast<float>(range_);

    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].z <= min_height_ || points[i].z >= max_height_) {
            map_idx_[i] = -1;
            continue;
        }
        // * the coordinates of x and y are exchanged here
        // (row <-> x, column <-> y)
        int pos_x = F2I(points[i].y, range_, inv_res_x);  // col
        int pos_y = F2I(points[i].x, range_, inv_res_y);  // row
        if (pos_x >= width_ || pos_x < 0 || pos_y >= height_ || pos_y < 0) {
            map_idx_[i] = -1;
            continue;
        }
        map_idx_[i] = pos_y * width_ + pos_x;

        int idx = map_idx_[i];
        float pz = points[i].z;
        // kitti dataset of intensify already in (0~1)!!!!!
        float pi = points[i].intensity / 255.0;
        // float pi = points[i].intensity;
        if (max_height_data_[idx] < pz) {
            max_height_data_[idx] = pz;
            top_intensity_data_[idx] = pi;
        }
        mean_height_data_[idx] += static_cast<float>(pz);
        mean_intensity_data_[idx] += static_cast<float>(pi);
        count_data_[idx] += float(1);

        total_points.at(idx).push_back(pz);

        num.at(idx)++;
        if(!init.at(idx)){
            min.at(idx) = pz;
            max.at(idx) = pz;
            init.at(idx) = true;
        }
        else{
            if(min.at(idx) > pz){
                min.at(idx) = pz;
            }
            if(max.at(idx) < pz){
                max.at(idx) = pz;
            }
        }
    }

    for (int i = 0; i < siz; ++i) {
        constexpr double EPS = 1e-6;
        if (count_data_[i] < EPS) {
            max_height_data_[i] = float(0);
        } else {
            mean_height_data_[i] /= count_data_[i];
            mean_intensity_data_[i] /= count_data_[i];
            nonempty_data_[i] = float(1);
        }
        count_data_[i] = logCount(static_cast<int>(count_data_[i]));

        bool extraction = false;
        float mid_pz = middle_point(total_points.at(i));
        float avg_pz = (max.at(i) + min.at(i))/2.0;

        if(fabs(mid_pz - avg_pz) <= 0.4){
            extraction = true;
        }

        // height Threshold 0.05 largest grid extract
        if(num.at(i) >= 1){
            if(max.at(i) - min.at(i) >= m_height_diff_threshold && extraction == true){
                height_map_.at(i) = true;
            }
            else{
                height_map_.at(i) = false;
            }
        }
    }
}
