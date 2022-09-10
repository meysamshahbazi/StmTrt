#ifndef __CALIBRATOR_HPP__
#define __CALIBRATOR_HPP__

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
#include <iterator>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;
#include <opencv2/imgproc.hpp>

using namespace cv;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum class calibration_model: int 
{
    BASE_Q      =   0,
    MEMORIZE    =   1,
};

class MyCalibrator: public nvinfer1::IInt8Calibrator
{
private:
    int32_t batch_size;
    std::string image_path;
    std::string calib_table_path;
    calibration_model cal_type;
    std::vector<char> calibration_cache;

    std::vector<std::string> img_list;
    const int64 img_size = 289*289;
    void * device_binind;
    void * device_binind_fg_bg;
    int img_index;
    bool read_cache{false};
    // this contans all img files pathes and bboxs
    std::vector< std::vector<std::string> > img_path_bb;
    
public: 
    MyCalibrator(int32_t batch_size,std::string image_path,std::string calib_table_path,calibration_model cal_type);
    ~MyCalibrator() = default;
    int32_t getBatchSize() const noexcept override ;
    virtual bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept;
    void getBatch_Q(void* bindings[], char const* names[], int32_t nbBindings) noexcept;
    void getBatch_M(void* bindings[], char const* names[], int32_t nbBindings) noexcept;
    virtual void const* readCalibrationCache(std::size_t& length) noexcept;
    virtual void writeCalibrationCache(void const* ptr, std::size_t length) noexcept;
    virtual nvinfer1::CalibrationAlgoType getAlgorithm() noexcept;

};

#endif // __CALIBRATOR_HPP__