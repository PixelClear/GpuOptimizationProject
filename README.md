# GpuOptimizationProject
This is a project that showcases GPU/parallel programming using parallel primitives like reduction. Starting from a naïve implementation we will go on making improvements and along the way, we will learn about GPU architecture and how to optimize/best practices. 

The main purpose behind this effort is to program and using profiling tools look at areas for improvement. I find very less examples where we see code profile dumps and explanations on how to make sense of it and find areas of improvement in code.


**<ins>Reduce_0 : 3.049 ms</ins>**

Uses strided pattern and shared memory. (add dig showing the pattern).

InputSize: 160 million ints

GPU: 6800W Pro 32GB

Theoretical memory bandwidth: 512 GB/s 

Achieved memory bandwidth: 209.90 GB/s
