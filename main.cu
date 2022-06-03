#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "core.cu"

// CUDA通用函数声明
void (*func)(int, int, int, float*, float*, int**);

// 计算两点间的距离
float cal_distance(int k, int idm, int idn, float *s_points, float *r_points) {
    float squareSum = 0;
    float diff;
    for (int i = 0; i < k; i++) {
        diff = s_points[k*idm+i] - r_points[k*idn+i];
        squareSum += (diff * diff);
    }
    return sqrt(squareSum);
}

// 生成随机数
float getRand() {
    return rand() / double(RAND_MAX);
}

// 生成随机样例
void getSample(int k, int m, int n, float **s_points, float **r_points) {
    float *tmp;
    tmp = (float*)malloc(sizeof(float) * k * m);
    for (int i = 0; i < k * m; i++) 
        tmp[i] = getRand();
    *s_points = tmp;
    tmp = (float*)malloc(sizeof(float) * k * n);
    for (int i = 0; i < k * n; i++) 
        tmp[i] = getRand();
    *r_points = tmp;
}

// 样例设置
int samples[] = {
    3,  1,      1024,
    3,  1,      65536,
    16, 1,      65536,

    3,  1024,   1024,
    3,  1024,   65536,
    16, 1024,   65536,

    3,  1,      16777216,
    16, 1,      16777216,

    3,  1024,   1048576,
    16, 1024,   1048576
};

int total = 0;      // 样例个数
int seed = 1000;    // 随机种子
long st, et;        // 开始和结束的时间

// 样例测试
void test(int v) {
    srand(seed);
    for (int i = 0; i < total; ++i) {
        int k = samples[3*i];
        int m = samples[3*i+1];
        int n = samples[3*i+2];
        float *s_points, *r_points;
        getSample(k, m, n, &s_points, &r_points);
        int *results;
        st = getTime();
        (*func)(k, m, n, s_points, r_points, &results);
        et = getTime();
        printf("CudaCall %d, %2d, %4d, %5d, %10.3fms\n", v, k, m, n, (et - st) / 1e6);
        free(s_points);
        free(r_points);
    }
}

int main() {
    total = sizeof(samples) / (3 * sizeof(*samples));   // 样例个数
    // 运行全部的优化版本
    for (int v = 1; v < 11; ++v) {
        switch (v) {
            case 0:
                func = & v0::cudaCall;
                break;
            case 1:
                func = & v1::cudaCall;
                break;
            case 2:
                func = & v2::cudaCall;
                break;
            case 3:
                func = & v3::cudaCall;
                break;
            case 4:
                func = & v4::cudaCall;
                break;
            case 5:
                func = & v5::cudaCall;
                break;
            case 6:
                func = & v6::cudaCall;
                break;
            case 7:
                func = & v7::cudaCall;
                break;
            case 8:
                func = & v8::cudaCall;
                break;
            case 9:
                func = & v9::cudaCall;
                break;
            case 10:
                func = & v10::cudaCall;
                break;
        }
        printf("\nRunning CUDACALL%d...\n", v);
        test(v);
    } 
}