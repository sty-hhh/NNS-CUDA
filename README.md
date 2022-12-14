# NNS-CUDA
Implement 14 different approaches to optimize *Nearest Neighbor Search*.
## Detail
- V0: CPU Linear Search
- V1: GPU Distance Matrix
- V2: GPU Thrust
- V3: GPU Shared Memory Tree Reduction
- V4: GPU AoS2SoA
- V5: GPU Texture Memory
- V6: GPU Constant Memory
- V7: GPU Multiblock
- V8: GPU Reduction of 4 GPUs
- V9: GPU Full Loop Expansion
- V10: CPU KD-Tree
- V11: GPU KD-Tree
- V12: CPU Octree
- V13: GPU Octree
## Run
~~~powershell
$ nvcc -Xcompiler -fopenmp -arch=sm_70 main.cu -o main
$ ./main
~~~
