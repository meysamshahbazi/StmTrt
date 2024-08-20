#include "cuproc.h"

CudaProcess::CudaProcess(cv::Mat im) :im{im}
{

}

CudaProcess::~CudaProcess()
{

}

void* CudaProcess::getImgPtr()
{
    cv::Mat im_bgra;
    cv::cvtColor(im, im_bgra, cv::COLOR_BGR2RGBA);

    uchar4 im_bgra_ptr = new uchar4[im_bgra]


    return im_ptr;
}

void CudaProcess::freeImage()
{
    cudaFree(im_ptr);
}



