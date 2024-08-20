#ifndef __STMTRACKER_HPP__
#define __STMTRACKER_HPP__
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <opencv2/core/types.hpp>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// #include "tracking_algorithm.h"
#include "trt_utils.h"
#include "cudaWarp.h"
#include "cudaResize.h"
#include "cudaCrop.h"

#include <opencv2/cudawarping.hpp>

using namespace std;
using namespace cv;

class stmtracker/*  : public TrackerAlgorithm */
{
private:
    float *data_q;
    float *data_m;
    float *fg_bg_label_map;
    float * box;
    float * score;
    
    cudaStream_t stream_m;
    Logger logger;
    const string  model_path_base_q{"../../eff-onnx/backbone_q.onnx"};
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_base_q{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_base_q{nullptr};
    const string  model_path_base_m{"../../eff-onnx/memorize.onnx"};
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_base_m{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_base_m{nullptr};
    const string  model_path_head{"../../eff-onnx/head.onnx"};
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_head{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_head{nullptr};

    // int32_t size_buf_base_q{2};
    std::vector<void *> buffers_base_q;
    std::vector<void *> buffers_base_m;
    std::vector<void *> buffers_head;

    std::vector<nvinfer1::Dims> input_dims_base_q;
    std::vector<nvinfer1::Dims> output_dims_base_q;
    std::vector<nvinfer1::Dims> input_dims_base_m;
    std::vector<nvinfer1::Dims> output_dims_base_m;
    std::vector<nvinfer1::Dims> input_dims_head;
    std::vector<nvinfer1::Dims> output_dims_head;

    uchar4 *im_m_crop_ptr;
    uchar4 *im_q_crop_ptr;

    cv::Point2f target_pos;
    cv::Size2f target_sz;
    
    const int min_w{10};
    const int min_h{10};

    const int score_size{25};
    Mat window;
    float* d_window{nullptr};
    std::vector<void *> all_memory_frame_feats; // TODO: change this
    Scalar avg_chans; // this has 4 value and the order is not the same as in python
    // Mat last_img;
    int last_fd;
    std::vector<float> pscores;
    int cur_frame_idx{0};
    const float search_area_factor{4.0};
    const int q_size{200};
    const int m_size{200};
    float target_scale;
    cv::Size2f base_target_sz;

    float window_influence{0.21};

    cv::Point2f target_pos_prior;
    cv::Size2f target_sz_prior;

    const int num_segments=4;
    float scale_q;
    float penalty_k=0.04;

    float test_lr{0.95};

    void create_trt_buffer();
    void load_trt_engines();
    
    std::vector<int> select_representatives(int cur_frame_idx);
    void _postprocess_score(float * score,const std::vector<std::vector<float>> &box_wh,std::vector<float> &pscore,std::vector<float> &penalty);
    float change(float r);
    float sz(float w,float h);
    void _postprocess_box(float score_best,std::vector<float> box_best,float penalty_best,cv::Point2f &new_target_pos,cv::Size2f &new_target_sz);
    void createWindow();
    std::vector<std::vector<float>> xyxy2cxywh(float *box);

    void get_crop_single(int fd, Point2f target_pos_,
                                float target_scale, float* blob, 
                                uchar4* im_patch, float &real_scale,
                                cudaStream_t stream = 0); 

    void get_crop_single2(int fd, Point2f target_pos_,
                                float target_scale, float* blob, 
                                uchar4* im_patch, float &real_scale,
                                cudaStream_t stream = 0); 


    void memorize();
    void track(int fd, std::vector<void *> &features,cv::Point2f &new_target_pos, cv::Size2f &new_target_sz);

    void create_fg_bg_label_map(float* fg_bg, cv::Rect2i &bb, cudaStream_t stream);

public:
    // stmtracker() = delete;
    stmtracker(/* OutVidConf_t vid_conf_ */);
    ~stmtracker();
    void init(cv::Mat im, const cv::Rect2f state);
    cv::Rect2f update(cv::Mat im);

    static void gen_engine_from_onnx();
    
};

#endif