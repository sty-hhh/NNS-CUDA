#include "time.h"

// divup calculates n / m and would round it up if the remainder is non-zero.
int divup(int n, int m) {
    return n % m == 0 ? n/m : n/m + 1;
}

// getTime gets the local time in nanoseconds.
long getTime() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

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

