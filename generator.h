#include <stdio.h>
#include <stdlib.h>

const double DOUBLE_RAND_MAX = double(RAND_MAX);

float getRandNum() {
    return rand() / DOUBLE_RAND_MAX;
}

void getSample(int k, int m, int n, float **searchPoints, float **referencePoints) {
    float *tmp;
    tmp = (float*)malloc(sizeof(float) * k * m);
    for (int i = 0; i < k * m; i++) 
        tmp[i] = getRandNum();
    *searchPoints = tmp;
    tmp = (float*)malloc(sizeof(float) * k * n);
    for (int i = 0; i < k * n; i++) 
        tmp[i] = getRandNum();
    *referencePoints = tmp;
}
