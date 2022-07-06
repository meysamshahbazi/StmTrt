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
    

public:
    stmtracker(/* args */);
    ~stmtracker();
    void init(Mat frame, Rect box);
    Rect update(Mat frame);
};




#endif