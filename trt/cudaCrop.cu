#include "cudaCrop.h"


__global__ void DownsampleGPU(uchar4* srcImage, uchar4* dstImage, size_t src_width, size_t src_height,
	int x0, int y0, int df, size_t dst_width, size_t dst_height)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if( x >= dst_width || y >= dst_height  )
		return; 

	const int pixel_src = (y*df+y0) * src_width + (x*df+x0);
	const int pixel_dst = y * dst_width + x;

	dstImage[pixel_dst] = srcImage[pixel_src];
}



cudaError_t cudaDownsample( uchar4* srcDev, uchar4* dstDev, size_t src_width, size_t src_height,
	int x0, int y0, int df, size_t dst_width, size_t dst_height, cudaStream_t stream)
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( src_width == 0 || src_height == 0)
		return cudaErrorInvalidValue;

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(dst_width,blockDim.x), iDivUp(dst_height,blockDim.y), 1);

	DownsampleGPU<<<gridDim, blockDim,0,stream>>>( srcDev, dstDev, src_width, src_height, x0, y0, df, dst_width, dst_height);
	
	return CUDA(cudaGetLastError());
}