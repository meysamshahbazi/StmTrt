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

#include "utils.hpp"

using namespace std;
using namespace cv;

class stmtracker
{
private:
    Logger logger;
    const string  model_path_base_q{"../../backbone_q.onnx"};
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_base_q{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_base_q{nullptr};
    const string  model_path_base_m{"../../memorize.onnx"};
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_base_m{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_base_m{nullptr};
    const string  model_path_head{"../../head.onnx"};
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_head{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_head{nullptr};
    // int32_t size_buf_base_q{2};
    vector<void *> buffers_base_q;
    vector<void *> buffers_base_m;
    vector<void *> buffers_head;

    vector<nvinfer1::Dims> input_dims_base_q;
    vector<nvinfer1::Dims> output_dims_base_q;
    vector<nvinfer1::Dims> input_dims_base_m;
    vector<nvinfer1::Dims> output_dims_base_m;
    vector<nvinfer1::Dims> input_dims_head;
    vector<nvinfer1::Dims> output_dims_head;

    Point2f target_pos;
    Size2f target_sz;
    
    int im_h;
    int im_w;

    const int score_size{25};
    Mat window;
    vector<void *> all_memory_frame_feats; // TODO: change this
    Scalar avg_chans;// this has 4 value and the order is not the same as in python
    Mat last_img;
    std::vector<float> pscores;
    int cur_frame_idx {0};
    const float search_area_factor {4.0};
    const float q_size{289.0};
    float target_scale;
    Size2f base_target_sz;

    Point2f target_pos_prior;
    Size2f target_sz_prior;
    

public:
    stmtracker(/* args */);
    ~stmtracker();
    void init(Mat im, Rect2f state);
    Rect update(Mat im);
    void memorize();
    void get_crop_single(const Mat &im,Point2f target_pos,float target_scale, // all input 
                        int output_sz, Scalar avg_chans, // all input 
                        Mat &im_patch, float &real_scale); // these are output 
    
};




#endif