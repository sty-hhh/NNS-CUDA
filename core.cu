#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "utils.h"
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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
	template <int BLOCK_DIM>
	static __global__ void get_min_kernel(		// 共享内存树形归约
		const int m,
		const int n,
		const float *__restrict__ dis,
		int *__restrict__  result)
	{
		const int id = blockIdx.x * gridDim.y + blockIdx.y;	
		if (id >= m)
			return;
		__shared__ float dis_s[BLOCK_DIM];
		__shared__ int ind_s[BLOCK_DIM];
		dis_s[threadIdx.x] = INFINITY;
		for (int idn = threadIdx.x + blockIdx.x * BLOCK_DIM; idn < n; idn += gridDim.x * BLOCK_DIM) {
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
		float *dis_d, *s_d, *r_d;
		int *results_d;
		CHECK(cudaMalloc((void**)&dis_d, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&s_d, k * m * sizeof(float)));
		CHECK(cudaMalloc((void**)&r_d, k * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&results_d, m * sizeof(int)));
		CHECK(cudaMemcpy((void*)s_d, (void*)s_points, k * m * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy((void*)r_d, (void*)r_points, k * n * sizeof(float), cudaMemcpyHostToDevice));
		const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;	// 设置blockSize
        get_dis_kernel<<<
			dim3(divup(n, BLOCK_DIM_X), divup(m, BLOCK_DIM_Y)),
			dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(k, m, n, s_d, r_d, dis_d);
		*results = (int *)malloc(sizeof(int) * m);
		const int BLOCK_DIM = 1024;		// 设置blockSize
		get_min_kernel<BLOCK_DIM><<<m, BLOCK_DIM>>> (m, n, dis_d, results_d);	// 计算最近邻点
		CHECK(cudaMemcpy((void**)*results, (void*)results_d, m * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK(cudaFree(dis_d));
		CHECK(cudaFree(s_d));
		CHECK(cudaFree(r_d));
		CHECK(cudaFree(results_d));
	}
}; 
// GPU: 使用thrust库
namespace v2
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
// GPU：计算距离并同时归约
namespace v3
{
	template <int BLOCK_DIM>
	static __global__ void cudaCallKernel(
		const int k,
		const int m,
		const int n,
		const float *__restrict__ s_points,
		const float *__restrict__ r_points,
		int *__restrict__ result)
	{
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= m)
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
		for (int offset = BLOCK_DIM >> 1; offset > 0; offset >>= 1) { // 树形归约
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
		float *s_d, *r_d;
		int *results_d;
		CHECK(cudaMalloc((void**)&s_d, k * m * sizeof(float)));
		CHECK(cudaMalloc((void**)&r_d, k * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&results_d, m * sizeof(int)));
		CHECK(cudaMemcpy((void*)s_d, (void*)s_points, k * m * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy((void*)r_d, (void*)r_points, k * n * sizeof(float), cudaMemcpyHostToDevice));
		*results = (int *)malloc(sizeof(int) * m);
		const int BLOCK_DIM = 1024;		// 设置blockSize
		cudaCallKernel<BLOCK_DIM><<<m, BLOCK_DIM>>> (k, m, n, s_d, r_d, results_d);	// 计算最近邻点
		CHECK(cudaMemcpy((void**)*results, (void*)results_d, m * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK(cudaFree(s_d));
		CHECK(cudaFree(r_d));
		CHECK(cudaFree(results_d));
	}
}; 
// GPU：AoS2SoA
namespace v4
{
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
		const float *__restrict__ s_points,
		const float *__restrict__ r_points,
		int *__restrict__ result) {
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= m)
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
		float *s_d, *r_d, *rr_d;
		int *results_d;
		CHECK(cudaMalloc((void**)&s_d, k * m * sizeof(float)));
		CHECK(cudaMalloc((void**)&r_d, k * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&rr_d, k * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&results_d, m * sizeof(int)));
		CHECK(cudaMemcpy((void*)s_d, (void*)s_points, k * m * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy((void*)r_d, (void*)r_points, k * n * sizeof(float), cudaMemcpyHostToDevice));

		const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
		mat_inv_kernel<<<
			dim3(divup(n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
			dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(k, n, r_d, rr_d);
		const int BLOCK_DIM = 1024;
		cudaCallKernel<BLOCK_DIM><<<m, BLOCK_DIM>>> (k, m, n, s_d, rr_d, results_d);	// 计算最近邻点
		
		CHECK(cudaMemcpy((void**)*results, (void*)results_d, m * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK(cudaFree(s_d));
		CHECK(cudaFree(r_d));
		CHECK(cudaFree(rr_d));
		CHECK(cudaFree(results_d));
    }
}; 
// GPU：使用纹理内存存储参考点集
namespace v5
{
	template <int BLOCK_DIM>
	static __global__ void cudaCallKernel(	// 计算距离并归约
		const int k,
		const int m,
		const int n,
		const float *__restrict__ s_points,
		cudaTextureObject_t texObj, //使用纹理对象
		int *__restrict__ result) {
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= m)
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

		float *s_d;
		int *results_d;
		CHECK(cudaMalloc((void**)&s_d, k * m * sizeof(float)));
		CHECK(cudaMalloc((void**)&results_d, m * sizeof(int)));
		CHECK(cudaMemcpy((void*)s_d, (void*)s_points, k * m * sizeof(float), cudaMemcpyHostToDevice));
		*results = (int *)malloc(sizeof(int) * m);
		
        const int BLOCK_DIM = 1024;
		cudaCallKernel<BLOCK_DIM><<<m, BLOCK_DIM>>> (k, m, n, s_d, texObj, results_d);	// 计算最近邻点

		// 销毁纹理对象
		CHECK(cudaDestroyTextureObject(texObj));

		CHECK(cudaMemcpy((void**)*results, (void*)results_d, m * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK(cudaFree(s_d));
		CHECK(cudaFree(results_d));
	}
}; 
// GPU：使用常量内存存储转置参考点集
namespace v6
{
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
		const float *__restrict__ r_points,
		int *__restrict__ result) {
		const int id = blockIdx.x * gridDim.y + blockIdx.y;
		if (id >= m)
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
		float *r_d, *rr_d;
		int *results_d;
		CHECK(cudaMalloc((void**)&r_d, k * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&rr_d, k * n * sizeof(float)));
		CHECK(cudaMalloc((void**)&results_d, m * sizeof(int)));
		CHECK(cudaMemcpy((void*)r_d, (void*)r_points, k * n * sizeof(float), cudaMemcpyHostToDevice));
		
		const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
		mat_inv_kernel<<<
			dim3(divup(n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
			dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(k, n, r_d, rr_d);
		const int BLOCK_DIM = 1024;
		cudaCallKernel<BLOCK_DIM><<<m, BLOCK_DIM>>> (k, m, n, rr_d, results_d);	// 计算最近邻点
		
		CHECK(cudaMemcpy((void**)*results, (void*)results_d, m * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK(cudaFree(r_d));
		CHECK(cudaFree(rr_d));
		CHECK(cudaFree(results_d));
	}
}; 
// GPU：多个block归约
namespace v7
{
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
        #pragma omp parallel num_threads(num_gpus) 	// 多卡并行
		{
			int thread_num = omp_get_thread_num();
			int thread_n = divup(n, num_gpus);
			float *thread_r_points = r_points + thread_num * thread_n * k;	// 为每张显卡分配定量的参考点集
			if (thread_num == num_gpus - 1) {
				thread_n = n - thread_num * thread_n;
				if (thread_n == 0) {
					thread_n = 1;
					thread_r_points -= k;
				}	
			}
			CHECK(cudaSetDevice(thread_num));	// 选择对应的显卡
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
            #pragma omp critical 	// 临界区将多卡结果合并
			{
				my_beg = results_tmp.size();
				results_tmp.insert(results_tmp.end(), results_d.begin(), results_d.end());
				my_end = results_tmp.size();
			}
            #pragma omp barrier		// 多卡同步
			for (int offset = (thread_r_points - r_points) / k; my_beg < my_end; ++my_beg)
					results_tmp[my_beg] += offset;	// 将每张卡上的参考点index转为全局index
		}
		*results = (int *)malloc(sizeof(int) * m);
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
// GPU：多卡规约+循环展开
namespace v9
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
		if (threadIdx.x < 512)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ 512]) {
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ 512];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ 512];
			}
		__syncthreads();
		if (threadIdx.x < 256)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ 256]) {
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ 256];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ 256];
			}
		__syncthreads();
		if (threadIdx.x < 128)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ 128]) {
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ 128];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ 128];
			}
		__syncthreads();
		if (threadIdx.x < 64)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ 64]) {
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ 64];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ 64];
			}
		__syncthreads();
		if (threadIdx.x < 32)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ 32]) {
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ 32];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ 32];
			}
		if (threadIdx.x < 16)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ 16]) {
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ 16];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ 16];
			}
		if (threadIdx.x < 8)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ 8]) {
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ 8];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ 8];
			}
		if (threadIdx.x < 4)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ 4]) {
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ 4];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ 4];
			}
		if (threadIdx.x < 2)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ 2]) {
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ 2];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ 2];
			}
		if (threadIdx.x == 0) 
			result[id] = dis_s[0] > dis_s[1] ? ind_s[1] : ind_s[0];	
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
        #pragma omp parallel num_threads(num_gpus) 	// 多卡并行
		{
			int thread_num = omp_get_thread_num();
			int thread_n = divup(n, num_gpus);
			float *thread_r_points = r_points + thread_num * thread_n * k;	// 为每张显卡分配定量的参考点集
			if (thread_num == num_gpus - 1) {
				thread_n = n - thread_num * thread_n;
				if (thread_n == 0) {
					thread_n = 1;
					thread_r_points -= k;
				}	
			}
			CHECK(cudaSetDevice(thread_num));	// 选择对应的显卡
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
            #pragma omp critical 	// 临界区将多卡结果合并
			{
				my_beg = results_tmp.size();
				results_tmp.insert(results_tmp.end(), results_d.begin(), results_d.end());
				my_end = results_tmp.size();
			}
            #pragma omp barrier		// 多卡同步
			for (int offset = (thread_r_points - r_points) / k; my_beg < my_end; ++my_beg)
					results_tmp[my_beg] += offset;	// 将每张卡上的参考点index转为全局index
		}
		*results = (int *)malloc(sizeof(int) * m);
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
// CPU :KDTree
// namespace v9
// {
// 	float *s_points, *r_points;
// 	int k;
// 	struct DimCmp {
// 		int dim;
// 		bool operator()(int lhs, int rhs) const
// 		{
// 			return r_points[lhs * k + dim] < r_points[rhs * k + dim];
// 		}
// 	};
// 	struct KDTreeCPU {
// 		thrust::host_vector<int> p, dim;
// 		KDTreeCPU(int n) : p(n << 2, -1), dim(p) {
// 			thrust::host_vector<int> se(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n));
// 			build(se.begin(), se.end());
// 		}
// 		void build(thrust::host_vector<int>::iterator beg, thrust::host_vector<int>::iterator end, int rt = 1) {
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
// 		thrust::pair<float, int> ask(int x, thrust::pair<float, int> ans = {INFINITY, 0}, int rt = 1) {
// 			if (dim[rt] < 0)
// 				return ans;
// 			float d = s_points[x * k + dim[rt]] - r_points[p[rt] * k + dim[rt]], tmp = 0;
// 			for (int idk = 0; idk < k; ++idk) {
// 				float diff = s_points[x * k + idk] - r_points[p[rt] * k + idk];
// 				tmp += diff * diff;
// 			}
// 			int w = d > 0;
// 			ans = ask(x, min(ans, {tmp, p[rt]}), (rt << 1) ^ w);
// 			if (ans.first > d * d - 1e-6)
// 				ans = ask(x, ans, (rt << 1) ^ w ^ 1);
// 			return ans;
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
// 		v9::k = k;
// 		v9::s_points = s_points;
// 		v9::r_points = r_points;
// 		KDTreeCPU kd(n);
// 		*results = (int *)malloc(sizeof(int) * m);
// 		for (int i = 0; i < m; ++i)
// 			(*results)[i] = kd.ask(i).second;
// 	}
// } 
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
// 		int prt = p[rt];
// 		if (prt < 0)
// 			return ans;
// 		float d = s_d[x * k + dimrt] - r_d[prt * k + dimrt], tmp = 0;
// 		for (int kInd = 0; kInd < k; ++kInd)
// 		{
// 			float diff = s_d[x * k + kInd] - r_d[prt * k + kInd];
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
// 	struct DimCmp
// 	{
// 		int dim;
// 		bool operator()(int lhs, int rhs) const
// 		{
// 			return r_points[lhs * k + dim] < r_points[rhs * k + dim];
// 		}
// 	};
// 	struct KDTreeGPU
// 	{
// 		thrust::host_vector<int> p, dim;
// 		thrust::device_vector<int> p_d, dim_d;
// 		thrust::device_vector<float> s_d, r_d;
// 		KDTreeGPU(int n, int m)
// 			: p(n << 2, -1),
// 			  dim(p),
// 			  s_d(s_points, s_points + k * m),
// 			  r_d(r_points, r_points + k * n)
// 		{
// 			thrust::host_vector<int> se(
// 				thrust::counting_iterator<int>(0),
// 				thrust::counting_iterator<int>(n));
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
// 			for (int kInd = 0; kInd < k; ++kInd)
// 			{
// 				float sum = 0, sa = 0;
// 				for (thrust::host_vector<int>::iterator it = beg; it != end; ++it)
// 				{
// 					float val = r_points[(*it) * k + kInd];
// 					sum += val, sa += val * val;
// 				}
// 				sa = (sa - sum * sum / (end - beg)) / (end - beg);
// 				if (sa_max < sa)
// 					sa_max = sa, dim[rt] = kInd;
// 			}
// 			thrust::host_vector<int>::iterator mid = beg + (end - beg) / 2;
// 			std::nth_element(beg, mid, end, DimCmp{dim[rt]});
// 			p[rt] = *mid;
// 			build(beg, mid, rt << 1);
// 			build(++mid, end, rt << 1 | 1);
// 		}
// 		void range_ask(int m, int *results)
// 		{
// 			thrust::device_vector<int> results_d(m);
// 			int minGridSize, blockSize;
// 			CHECK(cudaOccupancyMaxPotentialBlockSize(
// 				&minGridSize,
// 				&blockSize,
// 				range_ask_kernel));
// 			range_ask_kernel<<<
// 				divup(m, blockSize),
// 				blockSize>>>(
// 				thrust::raw_pointer_cast(s_d.data()),
// 				thrust::raw_pointer_cast(r_d.data()),
// 				thrust::raw_pointer_cast(dim_d.data()),
// 				thrust::raw_pointer_cast(p_d.data()),
// 				k,
// 				m,
// 				thrust::raw_pointer_cast(results_d.data()));
// 			thrust::copy(results_d.begin(), results_d.end(), results);
// 		}
// 	};
// 	static void cudaCall(
// 		int k,
// 		int m,
// 		int n,
// 		float *s_points,
// 		float *r_points,
// 		int **results)
// 	{
// 		if (k > 16)
// 			return v0::cudaCall(k, m, n, s_points, r_points, results);
// 		v10::k = k;
// 		v10::s_points = s_points;
// 		v10::r_points = r_points;
// 		KDTreeGPU kd(n, m);
// 		*results = (int *)malloc(sizeof(int) * m);
// 		printf("\n\n---\nsearch on KD-Tree: ");
// 		{
// 			//WuKTimer timer;
// 			kd.range_ask(m, *results);
// 		}
// 		printf("---\n\n");
// 	}
// } // namespace v10

struct WarmUP
{
	WarmUP(int k, int m, int n)
	{
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
		for (int i = 0; i < 10; ++i)
		{
			int *result;
			v9::cudaCall(k, m, n, s_points, r_points, &result);
			free(result);
		}
		free(s_points);
		free(r_points);
	}
};
static WarmUP warm_up(1, 1, 1 << 15);
