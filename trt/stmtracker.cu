#include <iostream>
#include <fstream>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <opencv2/core.hpp>
#include <algorithm>
#include <numeric>
#include <string>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
// #include <cudnn.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "cudaResize.h"
#include "cudaFilterMode.cuh"

#include "stmtracker.hpp"

#include <algorithm>
#include <array>

#include "cuproc.h"

#include <opencv2/highgui.hpp>





__global__ void gpuFillBlobRGBA_STM(uchar4* im_rgba, int model_sz, float* blob)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= model_sz || y >= model_sz)
        return;

    uchar4 pix = im_rgba[y*model_sz+x];

    blob[ y*model_sz + x + 2*model_sz*model_sz]  = static_cast<float>(pix.x);
    blob[ y*model_sz + x + 1*model_sz*model_sz]  = static_cast<float>(pix.y);
    blob[ y*model_sz + x + 0*model_sz*model_sz]  = static_cast<float>(pix.z);
    

    // B
    // blob[ y*model_sz + x + 0*model_sz*model_sz]  = static_cast<float>(im_rgba[y*model_sz+x].z);
    // // G
    // blob[ y*model_sz + x + 1*model_sz*model_sz]  = static_cast<float>(im_rgba[y*model_sz+x].y);
    // // R
    // blob[ y*model_sz + x + 2*model_sz*model_sz]  = static_cast<float>(im_rgba[y*model_sz+x].x);
}





void stmtracker::get_crop_single(cv::Mat fd, Point2f target_pos_,
                                float target_scale, float* blob, 
                                uchar4* im_patch, float &real_scale,
                                cudaStream_t stream) // these are output 
{
    // reversed pos!! 
    // pos is target_pos
    int output_sz = q_size; // = m_size

    Point2i posl = Point2i(target_pos_);
    float sample_sz = target_scale*output_sz;    
    float resize_factor = sample_sz/output_sz;
    
    int df = std::max(static_cast<int>(resize_factor-0.1),1);
    // Mat im2;
    uchar4* im2_ptr;
    CudaProcess cup{fd};
    int reduced_row_sz, reduced_col_sz;
    if (df > 1) {
        Point2i os = Point2i( posl.x % df, posl.y % df);
        posl = Point2i((posl.x-os.x)/df , (posl.y-os.y)/df);
        reduced_row_sz = (im_width - os.x+df-1)/df;
        reduced_col_sz = (im_hight - os.y+df-1)/df;
        cudaMalloc(&im2_ptr,reduced_row_sz*reduced_col_sz*4);
        void* img_ptr = cup.getImgPtr();

        cout << im_width << " , " << im_hight << reduced_row_sz << ", " << reduced_col_sz << std::endl;
        cudaError_t res = cudaDownsample((uchar4* )img_ptr, im2_ptr, im_width, im_hight,
	        os.x, os.y, df, reduced_row_sz, reduced_col_sz, stream); // TODO add Stream!
        
        if (res != cudaSuccess)
            std::cout << " *** Error in cudaDownsample " << res << endl;

        // int nRows = im.rows;
        // int nCols = im.cols * channels;
        // int i,j;
        // uchar* p;
        // std::vector<uchar> pixel_vec;
        
        // uchar img_data[reduced_row_sz*reduced_col_sz*3];
        // int data_idx = 0;
        // for( i = os.x; i < nRows; i+=df) {
        //     uchar* pixel = im.ptr<uchar>(i);  // point to first color in row
        //     for ( j = os.y*3; j < nCols;j+=3*df) {
        //         img_data[data_idx] = *pixel++;
        //         img_data[data_idx+1] = *pixel++;
        //         img_data[data_idx+2] = *pixel++;
        //         data_idx += 3;
        //         pixel += 3*df-3;
        //     }
        // }
        // im2 = cv::Mat(reduced_row_sz, reduced_col_sz, CV_8UC3,img_data);
    }
    else {
        im2_ptr = (uchar4*) cup.getImgPtr();
    }
    
    float sz = sample_sz/df;
    
    int szl = std::max(static_cast<int>(std::round(sz)) ,2);

    Point2i tl = Point2i(posl.x-(szl-1)/2,posl.y-(szl-1)/2);
    Point2i br = Point2i(posl.x+szl/2+1,posl.y+szl/2+1);

    float M_13 = tl.x;
    float M_23 = tl.y;
    float M_11 = (br.x-M_13)/(output_sz-1);
    float M_22 = (br.y-M_23)/(output_sz-1);

    float arr[2][3] = { 
                        {M_11,0,M_13},
                        {0,M_22,M_23}
                        };

    float m3x3[3][3] = { 
                        {M_11,0,M_13},
                        {0,M_22,M_23},
                        {0, 0, 1}
                        };

    // Mat mat2x3 = Mat(2,3, CV_32FC1 ,arr);


    // cudaWarpAffine(im2_ptr, im_patch, reduced_row_sz, reduced_col_sz,
    //                         arr,/* bool transform_inverted = */false,
    //                         /* cudaStream_t stream = */0 );

    cudaError_t res = cudaWarpPerspective(im2_ptr, reduced_row_sz, reduced_col_sz, IMAGE_RGBA8,
                                im_patch, output_sz, output_sz, IMAGE_RGBA8,
                                m3x3, /* bool transform_inverted */false, stream);
    if (res != cudaSuccess)
            std::cout << " ***Error in cudaWarpPerspective stram " << res << endl;
    // // cv::cuda::GpuMat()
    // // cv::cuda::warpAffine()

    // cv::warpAffine( im2,
    //                 im_patch,
    //                 mat2x3,
    //                 Size(output_sz,output_sz),
    //                 INTER_LINEAR | WARP_INVERSE_MAP,
    //                 BORDER_CONSTANT,
    //                 avg_chans
    // ); 	

    if (df > 1) 
        cudaFree(im2_ptr);

    const dim3 blockDim1(8, 8);
    const dim3 gridDim1(iDivUp(output_sz,blockDim1.x), iDivUp(output_sz,blockDim1.y));

    gpuFillBlobRGBA_STM<<<gridDim1, blockDim1, 0, stream>>>(im_patch, output_sz, blob);

    real_scale = static_cast<float>(output_sz)/((br.x-tl.x+1)*df); 

    // just x will be used!
    // real_scale.x = static_cast<float>(output_sz)/((br.x-tl.x+1)*df); 
    // real_scale.y = static_cast<float>(output_sz)/((br.y-tl.y+1)*df); 

    cup.freeImage();
}


__global__ void gpuCropFromRGBA_STM(uchar4* src, int xMin,int yMin,int xMax,int yMax,int dst_width, uchar4* dst_rgba,
    int src_width,int src_height, int src_pitch )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // const int src_width = 1920;
    // const int src_height = 1080;
    if ( x >= int(xMax - xMin + 1) || y >=int(yMax - yMin + 1) )
        return;

    if(xMin <0 || yMin <0 || xMax >= src_width || yMax >= src_height) {
        dst_rgba[y*dst_width+x] = make_uchar4(127,127,127,127);
    }
    else {
        dst_rgba[y*dst_width+x] = src[(y+yMin)*src_pitch+ x+xMin];
    }
}



void stmtracker::get_crop_single2(cv::Mat fd, Point2f target_pos_,
                                float target_scale, float* blob, 
                                uchar4* im_patch, float &real_scale,
                                cudaStream_t stream) // these are output 
{
    // reversed pos!! 
    // pos is target_pos
    int output_sz = q_size; // = m_size

    Point2i posl = Point2i(target_pos_);
    float sample_sz = target_scale*output_sz;    
    float resize_factor = sample_sz/output_sz;
    
    int df = std::max(static_cast<int>(resize_factor-0.1),1);
    // Mat im2;
    uchar4* im2_ptr;
    CudaProcess cup{fd};
    int reduced_row_sz, reduced_col_sz;
    std::cout << df << std::endl;
    if (df > 1) {
        Point2i os = Point2i( posl.x % df, posl.y % df);
        posl = Point2i((posl.x-os.x)/df , (posl.y-os.y)/df);
        reduced_row_sz = (im_width - os.x+df-1)/df;
        reduced_col_sz = (im_hight - os.y+df-1)/df;
        cudaMalloc(&im2_ptr,reduced_row_sz*reduced_col_sz*4);
        void* img_ptr = cup.getImgPtr();
        // cout << im_width << " , " << im_hight <<", " << reduced_row_sz << ", " << reduced_col_sz << std::endl;
        cudaError_t res = cudaDownsample((uchar4* )img_ptr, im2_ptr, im_width, im_hight,
	        os.x, os.y, df, reduced_row_sz, reduced_col_sz, stream); // TODO add Stream!
        
        
        if (res != cudaSuccess)
            std::cout << " *** Error in cudaDownsample " << res << endl;

        

    }
    else {
        im2_ptr = (uchar4*) cup.getImgPtr();
        reduced_row_sz = im_width;
        reduced_col_sz = im_hight;
    }
    
    
    float sz = sample_sz/df;
    
    int szl = std::max(static_cast<int>(std::round(sz)) ,2);

    Point2i tl = Point2i(posl.x-(szl-1)/2,posl.y-(szl-1)/2);
    Point2i br = Point2i(posl.x+szl/2+1,posl.y+szl/2+1);

    float M_13 = tl.x;
    float M_23 = tl.y;
    float M_11 = (br.x-M_13)/(output_sz-1);
    float M_22 = (br.y-M_23)/(output_sz-1);

    float m3x3[3][3] = { 
                        {M_11,0,M_13},
                        {0,M_22,M_23},
                        {0, 0, 1}
                        };


    int leftPad = (int)(fmax(0., -tl.x));
    int topPad = (int)(fmax(0., -tl.y));
    int rightPad = (int)(fmax(0., br.x - im_width + 1.0f));
    int bottomPad = (int)(fmax(0., br.y - im_hight + 1.0f));


    int xMin = tl.x + leftPad;
    int yMin = tl.y + topPad;

    int xMax = br.x + leftPad;
    int yMax = br.y + topPad;


    uchar4 *img_dst;
    size_t im_patch_size = ((int)(xMax - xMin + 1))*((int)(yMax - yMin + 1))*sizeof(uchar4);

    cudaMallocManaged((void **)&img_dst, im_patch_size);


    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(int(xMax - xMin + 1),blockDim.x), iDivUp( int(yMax - yMin + 1),blockDim.y));
    gpuCropFromRGBA_STM<<<gridDim, blockDim,0 ,stream>>>( (uchar4 *)im2_ptr, int(xMin),int(yMin),int(xMax),int(yMax), int(xMax - xMin + 1),
                                            (uchar4 *) img_dst, reduced_row_sz, reduced_col_sz, reduced_row_sz);


    // std::cout << (xMax - xMin + 1) << " , " << (yMax - yMin + 1) << std::endl;
    // uchar *frame_ptr = new uchar[(xMax - xMin + 1)*(yMax - yMin + 1)*4];
    // cudaMemcpy(frame_ptr, img_dst, (xMax - xMin + 1)*(yMax - yMin + 1)*4, cudaMemcpyDeviceToHost);
    // cv::Mat frame = cv::Mat((yMax - yMin + 1), (xMax - xMin + 1), CV_8UC4, frame_ptr);
    // cv::imshow("img_dst", frame);
    // cv::waitKey(0);


    cudaResize( (uchar4*) img_dst,(size_t)(xMax - xMin + 1), (int)(yMax - yMin + 1), 
    (uchar4*) im_patch, output_sz, output_sz, FILTER_POINT, stream);

    // Mat mat2x3 = Mat(2,3, CV_32FC1 ,arr);


    // cudaWarpAffine(im2_ptr, im_patch, reduced_row_sz, reduced_col_sz,
    //                         arr,/* bool transform_inverted = */false,
    //                         /* cudaStream_t stream = */0 );

    // cudaError_t res = cudaWarpPerspective(im2_ptr, reduced_row_sz, reduced_col_sz, IMAGE_RGBA8,
    //                             im_patch, output_sz, output_sz, IMAGE_RGBA8,
    //                             m3x3, /* bool transform_inverted */false, stream);
    // if (res != cudaSuccess)
    //         std::cout << " ***Error in cudaWarpPerspective stram " << res << endl;
    // // cv::cuda::GpuMat()
    // // cv::cuda::warpAffine()

    // cv::warpAffine( im2,
    //                 im_patch,
    //                 mat2x3,
    //                 Size(output_sz,output_sz),
    //                 INTER_LINEAR | WARP_INVERSE_MAP,
    //                 BORDER_CONSTANT,
    //                 avg_chans
    // ); 	

    if (df > 1) 
        cudaFree(im2_ptr);

    const dim3 blockDim1(8, 8);
    const dim3 gridDim1(iDivUp(output_sz,blockDim1.x), iDivUp(output_sz,blockDim1.y));

    gpuFillBlobRGBA_STM<<<gridDim1, blockDim1, 0, stream>>>(im_patch, output_sz, blob);

    real_scale = static_cast<float>(output_sz)/((br.x-tl.x+1)*df); 

    // just x will be used!
    // real_scale.x = static_cast<float>(output_sz)/((br.x-tl.x+1)*df); 
    // real_scale.y = static_cast<float>(output_sz)/((br.y-tl.y+1)*df); 
    cudaFree(img_dst);
    cup.freeImage();
}


__global__ void fill_m(float * fg_bg,int x1, int y1, int x2, int y2, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    if( x> x1-1 && x <x2+1 && y > y1-1 && y <y2+1 )
        fg_bg[y*width + x] = 1.0;
    else 
        fg_bg[y*width + x] = 0.0;
}

void stmtracker::create_fg_bg_label_map(float* fg_bg, cv::Rect2i &bb, cudaStream_t stream){
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(m_size,blockDim.x), iDivUp(m_size,blockDim.y));

    float bg_fg_h[m_size * m_size];
    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size ; j++) {
            if( j> bb.x-1 && j <bb.x + bb.width+1 && i> bb.y-1 && i <bb.y + bb.height+1 )
                bg_fg_h[m_size*i+j ] = 1.0;
            else 
                bg_fg_h[m_size*i+j ] = 0.0;
        }

    cudaMemcpy(fg_bg, bg_fg_h,m_size * m_size * sizeof(float), cudaMemcpyHostToDevice);

    // fill_m <<<gridDim,blockDim, 0, stream >>>(fg_bg, bb.x, bb.y, bb.x + bb.width, bb.y + bb.height, m_size, m_size);
}
