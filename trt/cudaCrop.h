#ifndef _CUDA_CROP_H__
#define _CUDA_CROP_H__

#include "cudaUtility.h"

cudaError_t cudaDownsample( uchar4* srcDev, uchar4* dstDev, size_t src_width, size_t src_height,
	int x0, int y0, int df, size_t dst_width, size_t dst_height, cudaStream_t stream);

#endif