#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>





// divup calculates n / m and would round it up if the remainder is non-zero.
extern int divup(int n, int m);

// CHECK macro from Grossman and McKercher, "Professional CUDA C Programming"
#define CHECK(call)                                       \
    {                                                     \
        const cudaError_t error = call;                   \
        if (error != cudaSuccess)                         \
        {                                                 \
            printf("Error: %s:%d, ", __FILE__, __LINE__); \
            printf("code:%d, reason: %s \n",              \
                   error, cudaGetErrorString(error));     \
            exit(1);                                      \
        }                                                 \
    }
