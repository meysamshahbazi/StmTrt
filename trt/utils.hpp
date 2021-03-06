#ifndef __UTILS_HPP__
#define __UTILS_HPP__
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

#include <opencv2/imgproc.hpp>

using namespace cv;


// #include <opencv2/core/cuda_stream_accessor.hpp>
using namespace std;


struct  TRTDestroy
{
    template<class T>
    void operator()(T* obj) const 
    {
        if (obj)
            delete obj;
            // obj->destroy();
    }
};

class Logger : public nvinfer1::ILogger
{
void log(Severity severity, const char* msg) noexcept override
{
    // suppress info-level messages
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}
};// logger;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims);


void parseOnnxModel(const string & model_path,
                    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context);

Mat get_hann_win(Size sz);

void postprocessResults(float * gpu_output,const nvinfer1::Dims &dims, int batch_size, std::string file_name);

std::vector<vector<float>> xyxy2cxywh(const std::vector<float> & box);

#endif