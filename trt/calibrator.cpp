
#include "calibrator.hpp"
#include "stmtracker.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

int32_t MyCalibrator::getBatchSize() const noexcept 
{
    return batch_size;
}

MyCalibrator::MyCalibrator( int32_t batch_size,std::string image_path, 
                            std::string calib_table_path,calibration_model cal_type)
    :batch_size{batch_size},image_path{image_path},calib_table_path{calib_table_path},cal_type{cal_type}
{
    // for (const auto & entry : fs::directory_iterator(image_path))
    //     img_list.push_back(entry.path());

    // img_list.resize(static_cast<int>(img_list.size()/batch_size)*batch_size );
    // std::random_shuffle(img_list.begin(),img_list.end(),[](int i){return rand()%i;});
    if(cal_type == calibration_model::BASE_Q)
    {
        cudaMalloc(&device_binind, batch_size*img_size*3*sizeof(float));
    }
    if(cal_type == calibration_model::MEMORIZE)
    {
        cudaMalloc(&device_binind, batch_size*img_size*3*sizeof(float));
        cudaMalloc(&device_binind_fg_bg, batch_size*img_size*1*sizeof(float));
    }

    img_index = 0;

    // ../../../uav123dl/uav123_random_frame.txt
    std::ifstream frame_file{"../../../uav123dl/uav123_random_frame.txt"};
    std::string line;
    
    while (getline(frame_file,line))
    {
        std::stringstream Stream{line};
        std::string element;
        std::vector<std::string> elements;
        
        while (std::getline(Stream,element,','))
        {
            elements.push_back(element);
        }

        if(elements.size() == 5)
        {
            img_path_bb.push_back(elements); 
        }
    }

    std::random_shuffle(img_path_bb.begin(),img_path_bb.end());
    
    /*for( const auto l : img_path_bb)
    {
        cv::Mat img = cv::imread(l.at(0));
        cv::Rect roi =  
        cv::Rect(std::atoi(l[1].c_str()), std::atoi(l[2].c_str()),
                        std::atoi(l[3].c_str()), std::atoi(l[4].c_str()));
        cv::rectangle(img, roi, Scalar(255, 0, 0), 1, 1);
        cv::imshow("frame",img);
        if (cv::waitKey(1000) == 27) return;
    } */

}


bool MyCalibrator::getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept
{
    // if we have already leaded all images 
    // we will return false in order to exit from image loading
    if(img_index + batch_size >= img_path_bb.size()) return false;

    if(cal_type == calibration_model::BASE_Q)
    {
        getBatch_Q(bindings, names, nbBindings);
    }
    if(cal_type == calibration_model::MEMORIZE)
    {
        getBatch_M(bindings, names, nbBindings);
    }

    return true;
}

void MyCalibrator::getBatch_Q(void* bindings[], char const* names[], int32_t nbBindings) noexcept
{
    int q_size = 289;
    cv::Mat im_q_crop;
    float scale_q;
    const float search_area_factor {4.0};
    float batch_blob[batch_size * 3 * q_size * q_size];
    int offset_index = 0;
    
    for(int in_bacth_index{0}; in_bacth_index<batch_size;in_bacth_index++,img_index++)
    {
        std::vector<std::string> l = img_path_bb.at(img_index); 
        cv::Mat img_q = cv::imread(l.at(0)); 
        cv::Scalar avg_chans = cv::mean(img_q);
        cv::Rect2d box = cv::Rect(std::atoi(l[1].c_str()), std::atoi(l[2].c_str()),
                        std::atoi(l[3].c_str()), std::atoi(l[4].c_str()));

        cv::Point2f target_pos = (box.tl() + box.br())/2;
        cv::Size2f target_sz = box.size();
        
        float search_area = search_area_factor*search_area_factor*target_sz.area();
        float target_scale = std::sqrt(search_area)/q_size;

        get_crop_single(img_q,target_pos,target_scale,q_size,avg_chans,im_q_crop,scale_q);
        int data_idx = 0;
        
        for (int i = 0; i < im_q_crop.rows; ++i)
        {
            uchar* pixel = im_q_crop.ptr<uchar>(i);  // point to first color in row
            for (int j = 0; j < im_q_crop.cols; ++j)
            {
                batch_blob[offset_index + data_idx] = *pixel++;
                batch_blob[offset_index + data_idx+q_size*q_size] = *pixel++;
                batch_blob[offset_index + data_idx+2*q_size*q_size] = *pixel++;
                data_idx++;
            }
        }

        offset_index += 3*q_size*q_size;
    }
    
    cudaMemcpy( device_binind,batch_blob,
                batch_size * 3 * q_size * q_size*sizeof(float),
                cudaMemcpyHostToDevice);
    
    bindings[0] = device_binind;
}

void MyCalibrator::getBatch_M(void* bindings[], char const* names[], int32_t nbBindings) noexcept
{
    int q_size = 289;
    cv::Mat im_q_crop;
    float real_scale;
    const float search_area_factor {4.0};
    float batch_blob[batch_size * 3 * q_size * q_size];
    float fg_bg_label_map[batch_size * 1 * q_size * q_size];
    int offset_index = 0;
    
    for(int in_bacth_index{0}; in_bacth_index<batch_size;in_bacth_index++,img_index++)
    {
        std::vector<std::string> l = img_path_bb.at(img_index); 
        cv::Mat img_q = cv::imread(l.at(0)); 
        cv::Scalar avg_chans = cv::mean(img_q);
        cv::Rect2d box = cv::Rect(std::atoi(l[1].c_str()), std::atoi(l[2].c_str()),
                        std::atoi(l[3].c_str()), std::atoi(l[4].c_str()));

        cv::Point2f target_pos = (box.tl() + box.br())/2;
        cv::Size2f target_sz = box.size();
        
        float search_area = search_area_factor*search_area_factor*target_sz.area();
        float target_scale = std::sqrt(search_area)/q_size;
        float scale_m = std::sqrt(target_sz.area()/target_sz.area()); // TODO: the seceon target size
        // must be first frame target size!! we set it target_sz for ease of work!!
        get_crop_single(img_q, target_pos, scale_m, q_size, avg_chans, im_q_crop, real_scale);
        int data_idx = 0;
        
        for (int i = 0; i < im_q_crop.rows; ++i)
        {
            uchar* pixel = im_q_crop.ptr<uchar>(i);  // point to first color in row
            for (int j = 0; j < im_q_crop.cols; ++j)
            {
                batch_blob[3*offset_index + data_idx] = *pixel++;
                batch_blob[3*offset_index + data_idx+q_size*q_size] = *pixel++;
                batch_blob[3*offset_index + data_idx+2*q_size*q_size] = *pixel++;
                data_idx++;
            }
        }

        int x1 = (q_size-1)/2 - target_sz.width*real_scale/2;
        int y1 = (q_size-1)/2 - target_sz.height*real_scale/2;
        int x2 = (q_size-1)/2 + target_sz.width*real_scale/2;
        int y2 = (q_size-1)/2 + target_sz.height*real_scale/2;

        
        for (int i = 0; i < q_size; i++)
            for (int j = 0; j < q_size ; j++)
            {
                if( j> x1-1 && j <x2+1 && i> y1-1 && i <y2+1 )
                    fg_bg_label_map[offset_index + q_size*i+j ] = 1.0;
                else 
                    fg_bg_label_map[offset_index + q_size*i+j ] = 0.0;
            }

        offset_index += q_size*q_size;
    }
    
    cudaMemcpy( device_binind,batch_blob,
                batch_size * 3 * q_size * q_size*sizeof(float),
                cudaMemcpyHostToDevice);

    cudaMemcpy( device_binind_fg_bg, fg_bg_label_map,
                batch_size * 1 * q_size * q_size*sizeof(float),
                cudaMemcpyHostToDevice);
    
    bindings[0] = device_binind;
    bindings[1] = device_binind_fg_bg;
}

void const* MyCalibrator::readCalibrationCache(std::size_t& length) noexcept
{
    void* output;
    calibration_cache.clear();
    assert(!calib_table_path.empty());
    std::ifstream input(calib_table_path, std::ios::binary);
    input >> std::noskipws;
    if (read_cache && input.good())
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                  std::back_inserter(calibration_cache));

    length = calibration_cache.size();
    if (length)
    {
        std::cout << "Using cached calibration table to build the engine" << std::endl;
        output = &calibration_cache[0];
    }

    else
    {
        std::cout << "New calibration table will be created to build the engine" << std::endl;
        output = nullptr;
    }

    return output;
}

void MyCalibrator::writeCalibrationCache(void const* ptr, std::size_t length) noexcept
{
    assert(!calib_table_path.empty());
    std::ofstream output(calib_table_path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(ptr), length);
    output.close();
    return;
}

nvinfer1::CalibrationAlgoType MyCalibrator::getAlgorithm() noexcept
{
    return nvinfer1::CalibrationAlgoType::kMINMAX_CALIBRATION;
}

