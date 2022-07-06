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
#include <opencv2/core/cuda_stream_accessor.hpp>


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
} logger;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}


std::vector<std::string> getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

//   preprocessImage(frame,(float *)buffers[0],input_dims[0]);
// void preprocessImage(const string &image_path,float * gpu_input,const nvinfer1::Dims &dims)
void preprocessImage(cv::Mat frame ,float * gpu_input,const nvinfer1::Dims &dims)
{
    

    auto input_width = dims.d[3];
    auto input_height = dims.d[2];
    auto channels = dims.d[1];
    std::cout<<"input_width "<<input_width<<"input_height "<<input_height<<std::endl;
    // std::cout<<" dims.d[2] "<< dims.d[2]<<" dims.d[1] "<< dims.d[1]<<" dims.d[0] "<< dims.d[0]<<" dims.d[3] "<< dims.d[3]<<endl;
    auto input_size = cv::Size(input_width,input_height);


    cv::resize(frame, frame, input_size, 0, 0, cv::INTER_NEAREST);

    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(frame);

    

    // cv::cuda::GpuMat resized;
    // cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST,cvstream);

    cv::cuda::GpuMat flt_image;
    gpu_frame.convertTo(flt_image,CV_32FC3,1.0f/255.0f);

    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // to tensor is doing by this way and very important

    vector<cv::cuda::GpuMat> chw;

    for(size_t i =0; i <channels; i++)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input+i*input_width*input_height));
    }
    cv::cuda::split(flt_image,chw);

}

void parseOnnxModel(const string & model_path,
                    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context)
{
    // first we create builder 
    unique_ptr<nvinfer1::IBuilder,TRTDestroy> builder{nvinfer1::createInferBuilder(logger)};
    // then define flag that is needed for creating network definitiopn 
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<nvinfer1::INetworkDefinition,TRTDestroy> network{builder->createNetworkV2(flag)};
    // then parse network 
    unique_ptr<nvonnxparser::IParser,TRTDestroy> parser{nvonnxparser::createParser(*network,logger)};
    // parse from file
    parser->parseFromFile(model_path.c_str(),static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    // lets create config file for engine 
    unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,1U<<24);
    // config->setMaxWorkspaceSize(1U<<30);

    // use fp16 if it is possible 

    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // setm max bach size as it is very importannt for trt
    builder->setMaxBatchSize(1);
    // create engine and excution context
    // unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    engine.reset(builder->buildEngineWithConfig(*network,*config));
    context.reset(engine->createExecutionContext());

    return;
}

void postprocessResults(float * gpu_output,const nvinfer1::Dims &dims, int batch_size)
{
    // auto classes = getClassNames("../imagenet_classes.txt");

    vector<float> cpu_output(getSizeByDim(dims)*batch_size);
    cudaMemcpy(cpu_output.data(),gpu_output,cpu_output.size()*sizeof(float),cudaMemcpyDeviceToHost);
    cout<<"size : "<<cpu_output.size()<<endl;
    // compute SoftMax 
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val){return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(),cpu_output.end(),0.0);
    // std::cout<<"sum: "<<sum<< std::endl;

     // find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(dims) * batch_size);
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    // while (cpu_output[indices[i]] / sum > 0.005)
    // {
    //     if (classes.size() > indices[i])
    //     {
    //         // std::cout << "class: " << classes[indices[i]] << " | ";
    //     }
    //     std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "\n";
    //     ++i;
    // }

}


int main(int argc, const char ** argv) 
{
    if (argc < 3)
    {
        std::cerr<<"usage: " << argv[0] << " model.onnx image.jpg"<<std::endl;
        return -1;
    }

    string model_path = argv[1];
    string image_path = argv[2];
    const int batch_size = 1;

    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context{nullptr};

    parseOnnxModel(model_path,engine,context);

    // get sizes of input and output and allocate memory required for input data and for output data
    vector<nvinfer1::Dims> input_dims;
    vector<nvinfer1::Dims> output_dims;

    vector<void *> buffers(engine->getNbBindings());
    // std::cout<<"engine->getNbBindings()"<<engine->getNbBindings()<<endl;
    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }

    std::cout<<"size of input_dims: "<<input_dims.size()<<std::endl;
    std::cout<<"size of input_dims: "<<output_dims.size()<<std::endl;
    #define INPUT_H 289
    #define INPUT_W 289

    static float data[1 * 3 * 289 * 289];
    
    // cv::Mat frame = cv::imread(image_path);
    // if (frame.empty())
    // {
    //     std::cerr << "Input image " << image_path << " load failed\n";
    //     return 0;
    // }
    
    cv::Mat img = cv::imread(image_path);
    auto input_size = cv::Size(INPUT_W,INPUT_H);
    auto start = std::chrono::system_clock::now();
    cv::resize(img, img, input_size, 0, 0, cv::INTER_LINEAR);
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }
 
    // preprocessImage(image_path,(float *)buffers[0],input_dims[0]);
    // preprocessImage(frame,(float *)buffers[0],input_dims[0]);
    
    // cv::cuda::Stream cvstream;

    // cvstream.cudaPtr() must return 
    // cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvstream); 	
    
    // preprocessImage(frame,(float *)buffers[0],input_dims[0],cvstream);

    // cudaStream_t stream;
    // cudaStreamCreate(&stream);



    // cudaMemcpyAsync(buffers[0], data, batch_size * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpy(buffers[0], data, batch_size * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice);
    
    


    
    // context->executeV2(buffers.data())
    
    // context->enqueue(batch_size,buffers.data(),stream,nullptr);
    context->enqueue(batch_size,buffers.data(),0,nullptr);

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    // CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    // context.enqueue(batchSize, buffers, stream, nullptr);
    // CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    
    
    // cudaStreamSynchronize(stream);
    auto end = std::chrono::system_clock::now();
    // Release stream and buffers
    // cudaStreamDestroy(stream);
    
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;

        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    
    // context->execute(batch_size,buffers.data());
    // context->execute(bat)
    postprocessResults((float *) buffers[1], output_dims[0], batch_size);

    for (void * buf : buffers)
    {
        cudaFree(buf);
    }

    return 0;
}

