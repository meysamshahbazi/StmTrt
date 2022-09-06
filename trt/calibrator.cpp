
#include "calibrator.hpp"
#include "stmtracker.hpp"



int32_t MyCalibrator::getBatchSize() const noexcept 
{
    return batch_size;
}

MyCalibrator::MyCalibrator(int32_t batch_size,std::string image_path)
    :batch_size{batch_size},image_path{image_path}
{
    for (const auto & entry : fs::directory_iterator(image_path))
        img_list.push_back(entry.path());

    img_list.resize(static_cast<int>(img_list.size()/batch_size)*batch_size );
    std::random_shuffle(img_list.begin(),img_list.end(),[](int i){return rand()%i;});
    cudaMalloc(&device_binind, batch_size*img_size*3*sizeof(float));
    img_index = 0;
}


bool MyCalibrator::getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept
{
    // if we have already leaded all images 
    // we will return false in order to exit from image loading
    if(img_index + batch_size >= img_list.size()) return false;
    
    // otherwise we can load next batch:

    float *batch_blob;
    int q_size = 289;
    cv::Mat im_q_crop;
    cv::Mat img_q = cv::imread(img_list.at(img_index)); //TODO: fix this index!
    cv::Scalar avg_chans = cv::mean(img_q);
    float scale_q;
    cv::Rect2d box = cv::Rect2d(0,0,img_q.cols,img_q.rows); 
    cv::Point2f target_pos = (box.tl() + box.br())/2;
    cv::Size2f target_sz = box.size();
    const float search_area_factor {4.0};
    float search_area = search_area_factor*search_area_factor*target_sz.area();
    float target_scale = std::sqrt(search_area)/q_size;

    get_crop_single(img_q,target_pos,target_scale,q_size,avg_chans,im_q_crop,scale_q);





    return true;
}
void const* MyCalibrator::readCalibrationCache(std::size_t& length) noexcept
{
    return nullptr;
}
void MyCalibrator::writeCalibrationCache(void const* ptr, std::size_t length) noexcept
{
    return;
}
nvinfer1::CalibrationAlgoType MyCalibrator::getAlgorithm() noexcept
{
    return nvinfer1::CalibrationAlgoType::kLEGACY_CALIBRATION;
}

