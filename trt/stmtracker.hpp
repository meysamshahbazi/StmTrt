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
#include "utils.hpp"
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;

class stmtracker
{
private:
    float *data_q;
    float *data_m;
    float *fg_bg_label_map;
    float * box;
    float * score;
    
    cudaStream_t stream_m;
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
    int min_w{10};
    int min_h{10};

    const int score_size{25};
    Mat window;
    vector<void *> all_memory_frame_feats; // TODO: change this
    Scalar avg_chans;// this has 4 value and the order is not the same as in python
    Mat last_img;
    std::vector<float> pscores;
    int cur_frame_idx {0};
    const float search_area_factor {4.0};
    const int q_size{289};
    const int m_size{289};
    float target_scale;
    Size2f base_target_sz;

    float window_influence{0.21};

    Point2f target_pos_prior;
    Size2f target_sz_prior;

    const int num_segments=4;
    float scale_q;
    float penalty_k=0.04;

    float test_lr{0.95};

    vector<int> select_representatives(int cur_frame_idx);
    void _postprocess_score(float * score,const vector<vector<float>> &box_wh,vector<float> &pscore,vector<float> &penalty);
    float change(float r);
    float sz(float w,float h);
    void _postprocess_box(float score_best,vector<float> box_best,float penalty_best,Point2f &new_target_pos,Size2f &new_target_sz);


public:
    stmtracker(/* args */);
    ~stmtracker();
    void init(Mat im, Rect2f state);
    Rect2f update(Mat im);
    void memorize();
    void track(Mat im,vector<void *> &features,Point2f &new_target_pos, Size2f &new_target_sz);
    void get_crop_single(Mat &im,Point2f target_pos,float target_scale, // all input 
                        int output_sz, Scalar avg_chans, // all input 
                        Mat &im_patch, float &real_scale); // these are output 
    
};




#endif