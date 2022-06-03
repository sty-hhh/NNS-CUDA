#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "core.cu"

const double DOUBLE_RAND_MAX = double(RAND_MAX);

float getRandNum() {
    return rand() / DOUBLE_RAND_MAX;
}

void getSample(int k, int m, int n, float **s_points, float **r_points) {
    float *tmp;
    tmp = (float*)malloc(sizeof(float) * k * m);
    for (int i = 0; i < k * m; i++) 
        tmp[i] = getRandNum();
    *s_points = tmp;
    tmp = (float*)malloc(sizeof(float) * k * n);
    for (int i = 0; i < k * n; i++) 
        tmp[i] = getRandNum();
    *r_points = tmp;
}

// func is a function pointer which is compatible with "cudaCall"s.
void (*func)(int, int, int, float*, float*, int**);

// calcDistance is for calculating the precise Euclidean distance.
float calcDistance(int k, int mInd, int nInd, float *s_points, float *r_points) {
    float squareSum = 0;
    float diff;
    for (int i = 0; i < k; i++) {
        diff = s_points[k*mInd+i] - r_points[k*nInd+i];
        squareSum += (diff * diff);
    }
    return sqrt(squareSum);
}

int samplesConfig[] = {
    3,  1,      2,
    3,  2,      8,

    3,  1,      1024,
    3,  1,      65536,
    16, 1,      65536,

    3,  1024,   1024,
    3,  1024,   65536,
    16, 1024,   65536,
};

int numSamples = 0;
int seed = 1000;
long st, et;
// int **baselineResults = NULL;

void test(int v) {
    srand(seed);
    for (int i = 0; i < numSamples; ++i) {
        int k = samplesConfig[3*i];
        int m = samplesConfig[3*i+1];
        int n = samplesConfig[3*i+2];
        float *s_points, *r_points;
        getSample(k, m, n, &s_points, &r_points);
        
        
        int *results;
        st = getTime();
        (*func)(k, m, n, s_points, r_points, &results);
        et = getTime();
        printf("CudaCall %d, %2d, %4d, %5d, %10.3fms\n", v, k, m, n, (et - st) / 1e6);

        // if (baselineResults[i] == NULL) 
        //         baselineResults[i] = results;
        // else { 
        //     int errors = 0;
        //     for (int j = 0; j < m; j++) {
        //         if (baselineResults[i][j] == results[j]) 
        //             continue;
        //         else {
        //             float d1 = calcDistance(k, j, baselineResults[i][j], s_points, r_points);
        //             float d2 = calcDistance(k, j, results[j], s_points, r_points);
        //             if (d1 - d2 < -1e-3 || d1 - d2 > 1e-3) 
        //                 errors++;
        //         }
        //     }
        //     printf("errors/total w.r.t. baseline: %d/%d\n\n", errors, m);
        //     free(results);
        // }

        // De-allocate the memory spaces.
        free(s_points);
        free(r_points);
    }
}

int main() {
    numSamples = sizeof(samplesConfig) / (3 * sizeof(*samplesConfig));
    // baselineResults = (int **)malloc(sizeof(int *) * numSamples);
    // for (int i = 0; i < numSamples; i++) 
    //     baselineResults[i] = NULL;
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
    
    // if (baselineResults != NULL) {
    //     for (int i = 0; i < numSamples; i++) 
    //         free(baselineResults[i]);
    //     free(baselineResults);
    // }
}