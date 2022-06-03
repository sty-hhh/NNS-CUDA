#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "utils.h"
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int temp=0;

// CPU 串行版本
namespace v0
{
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		int *tmp = (int *)malloc(sizeof(int) * m);
		for (int i = 0; i < m; ++i) {
			float minSum = INFINITY;
			int index = 0;
			for (int j = 0; j < n; ++j) {
				float tempSum = 0;
				for (int t = 0; t < k; ++t) {
					const float diff = s_points[i * k + t] - r_points[j * k + t];	
					tempSum += diff * diff;		// 计算距离
				}
				if (minSum > tempSum) {			// 找出最小点
					minSum = tempSum;
					index = j;
				}
			}
			tmp[i] = index;
		}
		*results = tmp;
	}
} 
// GPU: 先计算 m*n 的距离矩阵，再求最近邻点
namespace v1
{
	extern __global__ void get_dis_kernel(
		const int k,							// 空间维度
		const int m,							// 查询点数量
		const int n,							// 参考点数量
		const float *__restrict__ s_points,		// 查询点集
		const float *__restrict__ r_points,		// 参考点集
		float *__restrict__ dis)				// 最近邻点集
	{
		const int idn = threadIdx.x + blockIdx.x * blockDim.x;
		const int idm = threadIdx.y + blockIdx.y * blockDim.y;
		if (idn < n && idm < m) {
			float tempSum = 0;
			for (int idk = 0; idk < k; ++idk) {
				const float diff = s_points[idk + idm * k] - r_points[idk + idn * k];
				tempSum += diff * diff;
			}
			dis[idn + idm * n] = tempSum;		// 计算 m*n 的距离矩阵
		}
	}
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		thrust::device_vector<float> dis_d(m * n);
		thrust::device_vector<float> s_d(s_points, s_points + k * m);
		thrust::device_vector<float> r_d(r_points, r_points + k * n);
		const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;	// 设置blockSize
        get_dis_kernel<<<
			dim3(divup(n, BLOCK_DIM_X), divup(m, BLOCK_DIM_Y)),
			dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
			k, m, n,
			thrust::raw_pointer_cast(s_d.data()),
			thrust::raw_pointer_cast(r_d.data()),
			thrust::raw_pointer_cast(dis_d.data()));
		*results = (int *)malloc(sizeof(int) * m);
		for (int i = 0; i < m; ++i)	// 找出最近邻点
			(*results)[i] = thrust::min_element(dis_d.begin() + n * i, dis_d.begin() + n * i + n) - dis_d.begin() - n * i;
	}
}; 
// GPU：共享内存树形归约
namespace v2
{
	__global__ void static get_dis_kernel(
		const int k,							// 空间维度
		const int m,							// 查询点数量
		const int n,							// 参考点数量
		const float *__restrict__ s_points,		// 查询点集
		const float *__restrict__ r_points,		// 参考点集
		float *__restrict__ dis)				// 最近邻点集
	{
		const int idn = threadIdx.x + blockIdx.x * blockDim.x;
		const int idm = threadIdx.y + blockIdx.y * blockDim.y;
		if (idn < n && idm < m) {
			float tempSum = 0;
			for (int idk = 0; idk < k; ++idk) {
				const float diff = s_points[idk + idm * k] - r_points[idk + idn * k];
				tempSum += diff * diff;
			}
			dis[idn + idm * n] = tempSum;		// 计算 m*n 的距离矩阵
		}
	}
	template <int BLOCK_DIM>
	static __global__ void get_min_kernel(		// 共享内存树形归约
		const int result_size,
		const int m,
		const int n,
		const float *__restrict__ dis,
		int *__restrict__ result)
	{
		const int id = blockIdx.x * gridDim.y + blockIdx.y;	
		if (id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM];
		__shared__ int ind_s[BLOCK_DIM];
		dis_s[threadIdx.x] = INFINITY;
		for (int idn = threadIdx.x + blockIdx.x * BLOCK_DIM; idn < n; idn += gridDim.x * BLOCK_DIM) {	// 不确定什么意思？
			const float tempSum = dis[idn + blockIdx.y * n];
			if (dis_s[threadIdx.x] > tempSum) {		// 赋值到共享内存
				dis_s[threadIdx.x] = tempSum;
				ind_s[threadIdx.x] = idn;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM >> 1; offset > 0; offset >>= 1) {	// 树形归约
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset]) {
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[id] = ind_s[0];
	}
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		thrust::device_vector<float> dis_d(m * n);
		thrust::device_vector<float> s_d(s_points, s_points + k * m);
		thrust::device_vector<float> r_d(r_points, r_points + k * n);
		const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;

		get_dis_kernel<<<	// 计算距离矩阵
			dim3(divup(n, BLOCK_DIM_X), divup(m, BLOCK_DIM_Y)),
			dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
			k, m, n,
			thrust::raw_pointer_cast(s_d.data()),
		    thrust::raw_pointer_cast(r_d.data()),
			thrust::raw_pointer_cast(dis_d.data()));

		thrust::device_vector<int> results_d(m);
		const int BLOCK_DIM = 1024;		// 设置blockSize

		get_min_kernel<BLOCK_DIM><<<	// 计算最近邻点
			dim3(results_d.size() / m, m), BLOCK_DIM>>> (
			results_d.size(),
			m, n,
			thrust::raw_pointer_cast(dis_d.data()),
			thrust::raw_pointer_cast(results_d.data()));

		thrust::copy(		// device_vector复制回host端
			results_d.begin(),
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
	}
}; 
// GPU：计算距离并同时规约
namespace v3
{
	template <int BLOCK_DIM>
	static __global__ void cudaCallKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ s_points,
		const float *__restrict__ r_points,
		int *__restrict__ result)
	{
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM];
		__shared__ int ind_s[BLOCK_DIM];
		dis_s[threadIdx.x] = INFINITY;
		for (int idm = blockIdx.y, idn = threadIdx.x + blockIdx.x * BLOCK_DIM; idn < n; idn += gridDim.x * BLOCK_DIM) {
			float tempSum = 0;
			for (int idk = 0; idk < k; ++idk) {		// 计算距离
				const float diff = s_points[idk + idm * k] - r_points[idk + idn * k];
				tempSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > tempSum) {
				dis_s[threadIdx.x] = tempSum;
				ind_s[threadIdx.x] = idn;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM >> 1; offset > 0; offset >>= 1) { // 树形规约
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset]) {
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[id] = ind_s[0];
	}
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		thrust::device_vector<int> results_d(m);	
	    thrust::device_vector<float> s_d(s_points, s_points + k * m);
		thrust::device_vector<float> r_d(r_points, r_points + k * n);
		const int BLOCK_DIM = 1024;

		cudaCallKernel<BLOCK_DIM><<<	// 计算距离并求最近邻
			dim3(results_d.size() / m, m), BLOCK_DIM>>>(	
			k, m, n, results_d.size(),
			thrust::raw_pointer_cast(s_d.data()),
			thrust::raw_pointer_cast(r_d.data()),
			thrust::raw_pointer_cast(results_d.data()));
		
		thrust::copy(		// device_vector复制回host端
			results_d.begin(),
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
	}
}; 
// GPU：每个维度上的点坐标连续存储
namespace v4 {
	static __global__ void mat_inv_kernel(	// 转置矩阵
		const int k,
		const int n,
		const float *__restrict__ input,
		float *__restrict__ output) {
		const int idn = threadIdx.x + blockIdx.x * blockDim.x;
		const int idk = threadIdx.y + blockIdx.y * blockDim.y;
		if (idn < n && idk < k) {
			const float a = input[idn * k + idk];
			output[idn + idk * n] = a;
		}
	}
	template <int BLOCK_DIM>
	static __global__ void cudaCallKernel(	// 计算距离并归约
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ s_points,
		const float *__restrict__ r_points,
		int *__restrict__ result) {
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM];
		__shared__ int ind_s[BLOCK_DIM];
		dis_s[threadIdx.x] = INFINITY;
		for (int idm = blockIdx.y, idn = threadIdx.x + blockIdx.x * BLOCK_DIM; idn < n; idn += gridDim.x * BLOCK_DIM) {
			float tempSum = 0;
			for (int idk = 0; idk < k; ++idk) {
				const float diff = s_points[idk + idm * k] - r_points[idk * n + idn];
				tempSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > tempSum) {
				dis_s[threadIdx.x] = tempSum;
				ind_s[threadIdx.x] = idn;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM >> 1; offset > 0; offset >>= 1) {
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset]) {
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[id] = ind_s[0];
	}
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		thrust::device_vector<int> results_d(m);
		thrust::device_vector<float> s_d(s_points, s_points + k * m);
		thrust::device_vector<float> r_d(r_points, r_points + k * n);
		thrust::device_vector<float> rr_d(k * n);
		const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;

		mat_inv_kernel<<<
			dim3(divup(n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
			dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
			k, n,
			thrust::raw_pointer_cast(r_d.data()),
			thrust::raw_pointer_cast(rr_d.data()));

		const int BLOCK_DIM = 1024;
		cudaCallKernel<BLOCK_DIM><<<	// 计算距离并求最近邻
			dim3(results_d.size() / m, m), BLOCK_DIM>>>(	
			k, m, n,
			results_d.size(),
			thrust::raw_pointer_cast(s_d.data()),
			thrust::raw_pointer_cast(rr_d.data()),
			thrust::raw_pointer_cast(results_d.data()));

		thrust::copy(		// device_vector复制回host端
			results_d.begin(),
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
    }
}; 
// GPU：使用纹理内存存储参考点集
namespace v5 {
	template <int BLOCK_DIM>
	static __global__ void cudaCallKernel(	// 计算距离并归约
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ s_points,
		cudaTextureObject_t texObj, //使用纹理对象
		int *__restrict__ result) {
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM];
		__shared__ int ind_s[BLOCK_DIM];
		dis_s[threadIdx.x] = INFINITY;
		for (int idm = blockIdx.y, idn = threadIdx.x + blockIdx.x * BLOCK_DIM; idn < n; idn += gridDim.x * BLOCK_DIM) {
			float tempSum = 0;
			for (int idk = 0; idk < k; ++idk) {
				const float diff = s_points[idk + idm * k] - tex2D<float>(texObj, idk, idn);
				tempSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > tempSum) {
				dis_s[threadIdx.x] = tempSum;
				ind_s[threadIdx.x] = idn;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM >> 1; offset > 0; offset >>= 1) {
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset]) {
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[id] = ind_s[0];
	}
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		if (n > 65536) {	// 纹理内存最大限制
			v4::cudaCall(k, m, n, s_points, r_points, results);
			return;
		}
		cudaArray *cuArray;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		CHECK(cudaMallocArray(&cuArray, &channelDesc, k, n));
		CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, r_points, sizeof(float) * k, sizeof(float) * k, n, cudaMemcpyHostToDevice));

		// 绑定纹理到cudaArray上
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// 设置纹理为只读
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;

		// 创建纹理对象
		cudaTextureObject_t texObj = 0;
		CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

		thrust::device_vector<int> results_d(m);
		thrust::device_vector<float> s_d(s_points, s_points + k * m);
		
        const int BLOCK_DIM = 1024;
		cudaCallKernel<BLOCK_DIM><<<dim3(results_d.size() / m, m), BLOCK_DIM>>> (
			k, m, n,
			results_d.size(),
			thrust::raw_pointer_cast(s_d.data()),
			texObj,
			thrust::raw_pointer_cast(results_d.data()));

		thrust::copy(
			results_d.begin(),
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
	}
}; 
// GPU：使用纹理内存存储转置参考点集
namespace v6 {
	static __constant__ float const_mem[(64 << 10) / sizeof(float)];	// 常量内存最大限制
	static __global__ void mat_inv_kernel(	// 矩阵转置
		const int k,	
		const int n,
		const float *__restrict__ input,
		float *__restrict__ output) {
		const int idn = threadIdx.x + blockIdx.x * blockDim.x;
		const int idk = threadIdx.y + blockIdx.y * blockDim.y;
		if (idn < n && idk < k) {
			const float a = input[idn * k + idk];
			output[idn + idk * n] = a;
		}
	}
	template <int BLOCK_DIM>
	static __global__ void cudaCallKernel(	// 计算距离并归约
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ r_points,
		int *__restrict__ result) {
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM];
		__shared__ int ind_s[BLOCK_DIM];
		dis_s[threadIdx.x] = INFINITY;
		for (int idm = blockIdx.y, idn = threadIdx.x + blockIdx.x * BLOCK_DIM; idn < n; idn += gridDim.x * BLOCK_DIM) {
			float tempSum = 0;
			for (int idk = 0; idk < k; ++idk) {
				const float diff = const_mem[idk + idm * k] - r_points[idk * n + idn];
				tempSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > tempSum) {
				dis_s[threadIdx.x] = tempSum;
				ind_s[threadIdx.x] = idn;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM >> 1; offset > 0; offset >>= 1) {
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset]) {
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[id] = ind_s[0];
	}
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		if (k * m > (64 << 10) / sizeof(float)) {
			v4::cudaCall(k, m, n, s_points, r_points, results);
			return;
		}
		CHECK(cudaMemcpyToSymbol(const_mem, s_points, sizeof(float) * k * m));	// 拷贝搜索点集到常量内存
		thrust::device_vector<int> results_d(m);
		thrust::device_vector<float> r_d(k * n);
		thrust::device_vector<float> rr_d(r_points, r_points + k * n);
		const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;

		mat_inv_kernel<<<
			dim3(divup(n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
			dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
			k, n,
			thrust::raw_pointer_cast(rr_d.data()),
			thrust::raw_pointer_cast(r_d.data()));
		
		const int BLOCK_DIM = 1024;
		cudaCallKernel<BLOCK_DIM><<<
			dim3(results_d.size() / m, m), BLOCK_DIM>>>(
            k, m, n, results_d.size(),
			thrust::raw_pointer_cast(r_d.data()),
			thrust::raw_pointer_cast(results_d.data()));

		thrust::copy(
			results_d.begin(),
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
	}
}; 
// GPU：多个block归约
namespace v7 {
	static __global__ void mat_inv_kernel(	// 矩阵转置
		const int k,
		const int n,
		const float *__restrict__ input,
		float *__restrict__ output) {
		const int idn = threadIdx.x + blockIdx.x * blockDim.x;
		const int idk = threadIdx.y + blockIdx.y * blockDim.y;
		if (idn < n && idk < k) {
			const float a = input[idn * k + idk];
			output[idn + idk * n] = a;
		}
	}
	template <int BLOCK_DIM>
	static __global__ void cudaCallKernel(	// 计算距离并归约
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ s_points,
		const float *__restrict__ r_points,
		int *__restrict__ result) 
    {
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM];
		__shared__ int ind_s[BLOCK_DIM];
		dis_s[threadIdx.x] = INFINITY;
		ind_s[threadIdx.x] = 0;
		for (int idm = blockIdx.y, idn = threadIdx.x + blockIdx.x * BLOCK_DIM; idn < n; idn += gridDim.x * BLOCK_DIM) {
			float tempSum = 0;
			for (int idk = 0; idk < k; ++idk) {
				const float diff = s_points[idk + idm * k] - r_points[idk * n + idn];
				tempSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > tempSum) {
				dis_s[threadIdx.x] = tempSum;
				ind_s[threadIdx.x] = idn;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM >> 1; offset > 0; offset >>= 1) {
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset]) {
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[id] = ind_s[0];
	}
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		thrust::device_vector<float> s_d(s_points, s_points + k * m);
		thrust::device_vector<float> r_d(r_points, r_points + k * n);
        thrust::device_vector<float> rr_d(k * n);
		const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
		mat_inv_kernel<<<
			dim3(divup(n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
			dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
			k, n,
			thrust::raw_pointer_cast(r_d.data()),
			thrust::raw_pointer_cast(rr_d.data()));

		const int BLOCK_DIM = 1024;
		int numBlocks;
		CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(	// 获取启动的block数量
			&numBlocks,
			cudaCallKernel<BLOCK_DIM>,
			BLOCK_DIM,
			0));
		thrust::device_vector<int> results_d(m * divup(numBlocks, m));

		cudaCallKernel<BLOCK_DIM><<<dim3(results_d.size() / m, m), BLOCK_DIM>>>(
			k, m, n,
			results_d.size(),
			thrust::raw_pointer_cast(s_d.data()),
			thrust::raw_pointer_cast(rr_d.data()),
			thrust::raw_pointer_cast(results_d.data()));
		
		*results = (int *)malloc(sizeof(int) * m);
		if (results_d.size() == m) {
			thrust::copy(results_d.begin(), results_d.end(), *results);
			return;
		}
		thrust::host_vector<int> results_tmp(results_d);
		for (int idm = 0; idm < m; ++idm) {	// CPU端归约查找最近邻点
			float minSum = INFINITY;
			int index = 0;
			for (int i = 0; i < results_tmp.size(); i += m) {
				const int idn = results_tmp[i];
				float tempSum = 0;
				for (int idk = 0; idk < k; ++idk) {
					const float diff = s_points[k * idm + idk] - r_points[k * idn + idk];
					tempSum += diff * diff;
				}
				if (minSum > tempSum) {
					minSum = tempSum;
					index = idn;
				}
			}
			(*results)[idm] = index;
        }
	}
}; 
// GPU: 多卡归约
namespace v8
{
	static __global__ void mat_inv_kernel(	// 矩阵转置
		const int k,
		const int n,
		const float *__restrict__ input,
		float *__restrict__ output)
	{
		const int idn = threadIdx.x + blockIdx.x * blockDim.x;
		const int idk = threadIdx.y + blockIdx.y * blockDim.y;
		if (idn < n && idk < k) {
			const float a = input[idn * k + idk];
			output[idn + idk * n] = a;
		}
	}
	template <int BLOCK_DIM>
	static __global__ void cudaCallKernel(	// 计算距离并归约
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ s_points,
		const float *__restrict__ r_points,
		int *__restrict__ result)
	{
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM];
		__shared__ int ind_s[BLOCK_DIM];
		dis_s[threadIdx.x] = INFINITY;
		ind_s[threadIdx.x] = 0;
		for (int idm = blockIdx.y, idn = threadIdx.x + blockIdx.x * BLOCK_DIM; idn < n; idn += gridDim.x * BLOCK_DIM) {
			float tempSum = 0;
			for (int idk = 0; idk < k; ++idk) {
				const float diff = s_points[idk + idm * k] - r_points[idk * n + idn];
				tempSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > tempSum) {
				dis_s[threadIdx.x] = tempSum;
				ind_s[threadIdx.x] = idn;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM >> 1; offset > 0; offset >>= 1) {
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset]) {
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[id] = ind_s[0];
	}
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		thrust::host_vector<int> results_tmp;
		int num_gpus = 0;
		CHECK(cudaGetDeviceCount(&num_gpus));	// 获得显卡数
		if (num_gpus > n)
			num_gpus = n;
		if (num_gpus < 1)
			return v0::cudaCall(k, m, n, s_points, r_points, results);
		if (n <= thrust::min(1 << 18, m << 10))
			return v7::cudaCall(k, m, n, s_points, r_points, results);
        #pragma omp parallel num_threads(num_gpus) 
		{
			int thread_num = omp_get_thread_num();
			int thread_n = divup(n, num_gpus);
			float *thread_r_points = r_points + thread_num * thread_n * k;
			if (thread_num == num_gpus - 1) {
				thread_n = n - thread_num * thread_n;
				if (thread_n == 0) {
					thread_n = 1;
					thread_r_points -= k;
				}	
			}
			CHECK(cudaSetDevice(thread_num));
			thrust::device_vector<float> s_d(s_points, s_points + k * m);
			thrust::device_vector<float> r_d(thread_r_points, thread_r_points + k * thread_n);
			thrust::device_vector<float> rr_d(k * thread_n);
			const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;

			mat_inv_kernel<<<
				dim3(divup(thread_n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
				dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
				k, thread_n,
				thrust::raw_pointer_cast(r_d.data()),
				thrust::raw_pointer_cast(rr_d.data()));
			
			const int BLOCK_DIM = 1024;
			int numBlocks;
			CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&numBlocks,
				cudaCallKernel<BLOCK_DIM>,
				BLOCK_DIM,
				0));
			thrust::device_vector<int> results_d(m * divup(numBlocks, m));
			cudaCallKernel<BLOCK_DIM><<<dim3(results_d.size() / m, m), BLOCK_DIM>>>(
				k, m, thread_n,
				results_d.size(),
				thrust::raw_pointer_cast(s_d.data()),
				thrust::raw_pointer_cast(rr_d.data()),
				thrust::raw_pointer_cast(results_d.data()));
			
			int my_beg, my_end;
            #pragma omp critical 
			{
				my_beg = results_tmp.size();
				results_tmp.insert(results_tmp.end(), results_d.begin(), results_d.end());
				my_end = results_tmp.size();
			}
            #pragma omp barrier
			for (int offset = (thread_r_points - r_points) / k; my_beg < my_end; ++my_beg)
					results_tmp[my_beg] += offset;
			
		}
		*results = (int *)malloc(sizeof(int) * m);
		for (int idm = 0; idm < m; ++idm) {
            float minSum = INFINITY;
			int index = 0;
			for (int i = 0; i < results_tmp.size(); i += m) {
				const int idn = results_tmp[i];
				float tempSum = 0;
				for (int idk = 0; idk < k; ++idk) {
					const float diff = s_points[k * idm + idk] - r_points[k * idn + idk];
					tempSum += diff * diff;
				}
				if (minSum > tempSum) {
					minSum = tempSum;
					index = idn;
				}
			}
			(*results)[idm] = index;
		}
	}
}; 
// CPU :KDTree
namespace v9
{
	float *s_points, *r_points;
	int k;
	struct DimCmp {
		int dim;
		bool operator()(int lhs, int rhs) const
		{
			return r_points[lhs * k + dim] < r_points[rhs * k + dim];
		}
	};
	struct KDTreeCPU {
		thrust::host_vector<int> p, dim;
		KDTreeCPU(int n) : p(n << 2, -1), dim(p) {
			thrust::host_vector<int> se(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n));
			build(se.begin(), se.end());
		}
		void build(thrust::host_vector<int>::iterator beg, thrust::host_vector<int>::iterator end, int rt = 1) {
			if (beg >= end)
				return;
			float sa_max = -INFINITY;
			for (int idk = 0; idk < k; ++idk) {
				float sum = 0, sa = 0;
				for (thrust::host_vector<int>::iterator it = beg; it != end; ++it) {
					float val = r_points[(*it) * k + idk];
					sum += val, sa += val * val;
				}
				sa = (sa - sum * sum / (end - beg)) / (end - beg);
				if (sa_max < sa)
					sa_max = sa, dim[rt] = idk;
			}
			thrust::host_vector<int>::iterator mid = beg + (end - beg) / 2;
			std::nth_element(beg, mid, end, DimCmp{dim[rt]});
			p[rt] = *mid;
			build(beg, mid, rt << 1);
			build(++mid, end, rt << 1 | 1);
		}
		thrust::pair<float, int> ask(int x, thrust::pair<float, int> ans = {INFINITY, 0}, int rt = 1) {
			if (dim[rt] < 0)
				return ans;
			float d = s_points[x * k + dim[rt]] - r_points[p[rt] * k + dim[rt]], tmp = 0;
			for (int idk = 0; idk < k; ++idk) {
				float diff = s_points[x * k + idk] - r_points[p[rt] * k + idk];
				tmp += diff * diff;
			}
			int w = d > 0;
			ans = ask(x, min(ans, {tmp, p[rt]}), (rt << 1) ^ w);
			if (ans.first > d * d - 1e-6)
				ans = ask(x, ans, (rt << 1) ^ w ^ 1);
			return ans;
		}
	};
	extern void cudaCall(
		int k, 				// 空间维度
		int m, 				// 查询点数量
		int n, 				// 参考点数量
		float *s_points, 	// 查询点集
		float *r_points, 	// 参考点集
		int **results)		// 最近邻点集
	{
		if (k > 16)
			return v0::cudaCall(k, m, n, s_points, r_points, results);
		v9::k = k;
		v9::s_points = s_points;
		v9::r_points = r_points;
		KDTreeCPU kd(n);
		*results = (int *)malloc(sizeof(int) * m);
		for (int i = 0; i < m; ++i)
			(*results)[i] = kd.ask(i).second;
	}
} 
// GPU :KDTree
// namespace v10
// {
// 	__device__ thrust::pair<float, int> ask_device(
// 		float *s_d,
// 		float *r_d,
// 		int *dim,
// 		int *p,
// 		int k,
// 		int x,
// 		thrust::pair<float, int> ans = {INFINITY, 0},
// 		int rt = 1)
// 	{
// 		int dimrt = dim[rt];
// 		if (dimrt < 0)
// 			return ans;
// 		int prt = p[rt];`
// 		if (prt < 0)
// 			return ans;
// 		float d = s_d[x * k + dimrt] - r_d[prt * k + dimrt], tmp = 0;
// 		for (int idk = 0; idk < k; ++idk) {
// 			float diff = s_d[x * k + idk] - r_d[prt * k + idk];
// 			tmp += diff * diff;
// 		}
// 		int w = d > 0;
// 		ans = ask_device(s_d, r_d, dim, p, k, x, thrust::min(ans, {tmp, prt}), (rt << 1) ^ w);
// 		if (ans.first > d * d - 1e-6)
// 			ans = ask_device(s_d, r_d, dim, p, k, x, ans, (rt << 1) ^ w ^ 1);
// 		return ans;
// 	}
// 	__global__ void range_ask_kernel(
// 		float *s_d,
// 		float *r_d,
// 		int *dim,
// 		int *p,
// 		int k,
// 		int m,
// 		int *results)
// 	{
// 		int global_id = blockIdx.x * blockDim.x + threadIdx.x;
// 		if (global_id >= m)
// 			return;
// 		results[global_id] = ask_device(s_d, r_d, dim, p, k, global_id).second;
// 	}
// 	float *s_points, *r_points;
// 	int k;
// 	struct DimCmp {
// 		int dim;
// 		bool operator()(int lhs, int rhs) const {
// 			return r_points[lhs * k + dim] < r_points[rhs * k + dim];
// 		}
// 	};
// 	struct KDTreeGPU {
// 		thrust::host_vector<int> p, dim;
// 		thrust::device_vector<int> p_d, dim_d;
// 		thrust::device_vector<float> s_d, r_d;
// 		KDTreeGPU(int n, int m) : 
//             p(n << 2, -1), dim(p), 
//             s_d(s_points, s_points + k * m),
// 			r_d(r_points, r_points + k * n) 
//         {
// 			thrust::host_vector<int> se(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n));
// 			build(se.begin(), se.end());
// 			dim_d = dim, p_d = p;
// 		}
// 		void build(
// 			thrust::host_vector<int>::iterator beg,
// 			thrust::host_vector<int>::iterator end,
// 			int rt = 1)
// 		{
// 			if (beg >= end)
// 				return;
// 			float sa_max = -INFINITY;
// 			for (int idk = 0; idk < k; ++idk) {
// 				float sum = 0, sa = 0;
// 				for (thrust::host_vector<int>::iterator it = beg; it != end; ++it) {
// 					float val = r_points[(*it) * k + idk];
// 					sum += val, sa += val * val;
// 				}
// 				sa = (sa - sum * sum / (end - beg)) / (end - beg);
// 				if (sa_max < sa)
// 					sa_max = sa, dim[rt] = idk;
// 			}
// 			thrust::host_vector<int>::iterator mid = beg + (end - beg) / 2;
// 			std::nth_element(beg, mid, end, DimCmp{dim[rt]});
// 			p[rt] = *mid;
// 			build(beg, mid, rt << 1);
// 			build(++mid, end, rt << 1 | 1);
// 		}
// 		void range_ask(int m, int *results) {
// 			thrust::device_vector<int> results_d(m);
// 			int minGridSize, blockSize;
// 			CHECK(cudaOccupancyMaxPotentialBlockSize(
// 				&minGridSize,
// 				&blockSize,
// 				range_ask_kernel));
// 			range_ask_kernel<<<divup(m, blockSize), blockSize>>> (
// 				thrust::raw_pointer_cast(s_d.data()),
// 				thrust::raw_pointer_cast(r_d.data()),
// 				thrust::raw_pointer_cast(dim_d.data()),
// 				thrust::raw_pointer_cast(p_d.data()), 
// 				k, m,
// 				thrust::raw_pointer_cast(results_d.data()));
// 			thrust::copy(results_d.begin(), results_d.end(), results);
// 		}
// 	};
// 	extern void cudaCall(
// 		int k, 				// 空间维度
// 		int m, 				// 查询点数量
// 		int n, 				// 参考点数量
// 		float *s_points, 	// 查询点集
// 		float *r_points, 	// 参考点集
// 		int **results)		// 最近邻点集
// 	{
// 		if (k > 16)
// 			return v0::cudaCall(k, m, n, s_points, r_points, results);
// 		v10::k = k;
// 		v10::s_points = s_points;
// 		v10::r_points = r_points;
// 		KDTreeGPU kd(n, m);
// 		*results = (int *)malloc(sizeof(int) * m);
// 		kd.range_ask(m, *results);
// 	}
// } 
namespace v10
{
	__device__ thrust::pair<float, int> ask_device(
		float *s_d,
		float *r_d,
		int *dim,
		int *p,
		int k,
		int x,
		thrust::pair<float, int> ans = {INFINITY, 0},
		int rt = 1)
	{
		int dimrt = dim[rt];
		if (dimrt < 0)
			return ans;
		int prt = p[rt];
		if (prt < 0)
			return ans;
		float d = s_d[x * k + dimrt] - r_d[prt * k + dimrt], tmp = 0;
		for (int kInd = 0; kInd < k; ++kInd)
		{
			float diff = s_d[x * k + kInd] - r_d[prt * k + kInd];
			tmp += diff * diff;
		}
		int w = d > 0;
		ans = ask_device(s_d, r_d, dim, p, k, x, thrust::min(ans, {tmp, prt}), (rt << 1) ^ w);
		if (ans.first > d * d - 1e-6)
			ans = ask_device(s_d, r_d, dim, p, k, x, ans, (rt << 1) ^ w ^ 1);
		return ans;
	}
	__global__ void range_ask_kernel(
		float *s_d,
		float *r_d,
		int *dim,
		int *p,
		int k,
		int m,
		int *results)
	{
		int global_id = blockIdx.x * blockDim.x + threadIdx.x;
		if (global_id >= m)
			return;
		results[global_id] = ask_device(s_d, r_d, dim, p, k, global_id).second;
	}
	float *s_points, *r_points;
	int k;
	struct DimCmp
	{
		int dim;
		bool operator()(int lhs, int rhs) const
		{
			return r_points[lhs * k + dim] < r_points[rhs * k + dim];
		}
	};
	struct KDTreeGPU
	{
		thrust::host_vector<int> p, dim;
		thrust::device_vector<int> p_d, dim_d;
		thrust::device_vector<float> s_d, r_d;
		KDTreeGPU(int n, int m)
			: p(n << 2, -1),
			  dim(p),
			  s_d(s_points, s_points + k * m),
			  r_d(r_points, r_points + k * n)
		{
			thrust::host_vector<int> se(
				thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>(n));
			build(se.begin(), se.end());
			dim_d = dim, p_d = p;
		}
		void build(
			thrust::host_vector<int>::iterator beg,
			thrust::host_vector<int>::iterator end,
			int rt = 1)
		{
			if (beg >= end)
				return;
			float sa_max = -INFINITY;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				float sum = 0, sa = 0;
				for (thrust::host_vector<int>::iterator it = beg; it != end; ++it)
				{
					float val = r_points[(*it) * k + kInd];
					sum += val, sa += val * val;
				}
				sa = (sa - sum * sum / (end - beg)) / (end - beg);
				if (sa_max < sa)
					sa_max = sa, dim[rt] = kInd;
			}
			thrust::host_vector<int>::iterator mid = beg + (end - beg) / 2;
			std::nth_element(beg, mid, end, DimCmp{dim[rt]});
			p[rt] = *mid;
			build(beg, mid, rt << 1);
			build(++mid, end, rt << 1 | 1);
		}
		void range_ask(int m, int *results)
		{
			thrust::device_vector<int> results_d(m);
			int minGridSize, blockSize;
			CHECK(cudaOccupancyMaxPotentialBlockSize(
				&minGridSize,
				&blockSize,
				range_ask_kernel));
			range_ask_kernel<<<
				divup(m, blockSize),
				blockSize>>>(
				thrust::raw_pointer_cast(s_d.data()),
				thrust::raw_pointer_cast(r_d.data()),
				thrust::raw_pointer_cast(dim_d.data()),
				thrust::raw_pointer_cast(p_d.data()),
				k,
				m,
				thrust::raw_pointer_cast(results_d.data()));
			thrust::copy(results_d.begin(), results_d.end(), results);
		}
	};
	static void cudaCall(
		int k,
		int m,
		int n,
		float *s_points,
		float *r_points,
		int **results)
	{
		if (k > 16)
			return v0::cudaCall(k, m, n, s_points, r_points, results);
		v10::k = k;
		v10::s_points = s_points;
		v10::r_points = r_points;
		KDTreeGPU kd(n, m);
		*results = (int *)malloc(sizeof(int) * m);
		printf("\n\n---\nsearch on KD-Tree: ");
		{
			//WuKTimer timer;
			kd.range_ask(m, *results);
		}
		printf("---\n\n");
	}
} // namespace v10
struct WarmUP
{
	WarmUP(int k, int m, int n)
	{
		void (*cudaCall[])(int, int, int, float *, float *, int **) = {
			v0::cudaCall,
			v1::cudaCall,
			v2::cudaCall,
			v3::cudaCall,
			v4::cudaCall,
			v5::cudaCall,
			v6::cudaCall,
			v7::cudaCall,
			v8::cudaCall};
		float *s_points = (float *)malloc(sizeof(float) * k * m);
		float *r_points = (float *)malloc(sizeof(float) * k * n);
		#pragma omp parallel
		{
			unsigned int seed = omp_get_thread_num(); //每个线程使用不同的随机数种子
			#pragma omp for
			for (int i = 0; i < k * m; ++i)
				s_points[i] = rand_r(&seed) / double(RAND_MAX); //使用线程安全的随机数函数
			#pragma omp for
			for (int i = 0; i < k * n; ++i)
				r_points[i] = rand_r(&seed) / double(RAND_MAX);
		}
		for (int i = 0; i < sizeof(cudaCall) / sizeof(cudaCall[0]); ++i)
		{
			int *result;
			cudaCall[i](k, m, n, s_points, r_points, &result);
			free(result);
		}
		free(s_points);
		free(r_points);
	}
};
static WarmUP warm_up(1, 1, 1 << 15);
