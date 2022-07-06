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
            obj->destroy();
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
// {
//     size_t size = 1;
//     for (size_t i = 0; i < dims.nbDims; ++i)
//     {
//         size *= dims.d[i];
//     }
//     return size;
// }

void parseOnnxModel(const string & model_path,
                    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context);
// {
//     Logger logger;
//     // first we create builder 
//     unique_ptr<nvinfer1::IBuilder,TRTDestroy> builder{nvinfer1::createInferBuilder(logger)};
//     // then define flag that is needed for creating network definitiopn 
//     uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//     unique_ptr<nvinfer1::INetworkDefinition,TRTDestroy> network{builder->createNetworkV2(flag)};
//     // then parse network 
//     unique_ptr<nvonnxparser::IParser,TRTDestroy> parser{nvonnxparser::createParser(*network,logger)};
//     // parse from file
//     parser->parseFromFile(model_path.c_str(),static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
//     for (int32_t i = 0; i < parser->getNbErrors(); ++i)
//     {
//         std::cout << parser->getError(i)->desc() << std::endl;
//     }
//     // lets create config file for engine 
//     unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};

//     config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,1U<<24);
//     // config->setMaxWorkspaceSize(1U<<30);

//     // use fp16 if it is possible 

//     if (builder->platformHasFastFp16())
//     {
//         config->setFlag(nvinfer1::BuilderFlag::kFP16);
//     }
//     // setm max bach size as it is very importannt for trt
//     builder->setMaxBatchSize(1);
//     // create engine and excution context
//     // unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
//     engine.reset(builder->buildEngineWithConfig(*network,*config));
//     context.reset(engine->createExecutionContext());

//     return;
// }

Mat get_hann_win(Size sz);
// {
//     Mat hann_rows = Mat::ones(sz.height, 1, CV_32F);
//     Mat hann_cols = Mat::ones(1, sz.width, CV_32F);
//     int NN = sz.height - 1;
//     if(NN != 0) {
//         for (int i = 0; i < hann_rows.rows; ++i) {
//             hann_rows.at<float>(i,0) = (float)(1.0/2.0 * (1.0 - cos(2*CV_PI*i/NN)));
//         }
//     }
//     NN = sz.width - 1;
//     if(NN != 0) {
//         for (int i = 0; i < hann_cols.cols; ++i) {
//             hann_cols.at<float>(0,i) = (float)(1.0/2.0 * (1.0 - cos(2*CV_PI*i/NN)));
//         }
//     }
//     return hann_rows * hann_cols;
// }


#endif