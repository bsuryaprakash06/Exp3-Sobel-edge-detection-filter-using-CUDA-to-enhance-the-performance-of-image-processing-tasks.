# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

| Field | Details |
|---|---|
| Name | Surya Prakash B |
| Reg No | 212224230281 |
| Experiment Number | 03 |
| Date | 22-05-2025 |

<h1 align=center> Sobel edge detection filter using CUDA </h1>
<h2 align=center>Implement Sobel edge detection filtern using GPU.</h2>


# Experiment Details:  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:

```cpp
%%writefile sobelEdgeDetectionFilter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__constant__ int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int Gy[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

__global__ void sobelFilter(unsigned char *srcImage,
                            unsigned char *dstImage,
                            unsigned int width,
                            unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 &&
        y >= 1 && y < height - 1) {

        int sumX = 0;
        int sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {

                unsigned char pixel =
                    srcImage[(y + i) * width + (x + j)];

                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        float magnitude =
            sqrtf((float)(sumX * sumX + sumY * sumY));

        magnitude = min(max((int)magnitude, 0), 255);

        dstImage[y * width + x] =
            (unsigned char)magnitude;
    }

    else if (x < width && y < height) {

        dstImage[y * width + x] = 0;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Read input image
    Mat image = imread("/content/WhiteHouse.jpg", IMREAD_COLOR);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    // Convert to grayscale
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    int width = gray_image.cols;
    int height = gray_image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate host memory for output image
    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, gray_image.data, imageSize, cudaMemcpyHostToDevice));

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    // Write output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);

    // Free memory
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print elapsed time
    printf("Total time taken: %f milliseconds\n", milliseconds);

    return 0;
}
```

## OUTPUT:

<table>
  <tr>
    <td>
      <img width="612" height="372" alt="WhiteHouse" src="https://github.com/user-attachments/assets/dd34eed9-9d45-4ad0-b46a-b72bcdade083" />
    </td>
    <td>
      <img width="515" height="372" alt="image" src="https://github.com/user-attachments/assets/407a94e0-1990-4691-a89c-681b3ad629b1" />
    </td>
  </tr>
</table>

## RESULT:
Thus the program has been executed by using CUDA to perform Sobel Edge Detection on an image successfully.

---

# Questions

## 1. What challenges did you face while implementing the Sobel filter for color images?

- Color images contain 3 channels (RGB), making memory handling more complex.
- Applying Sobel separately on each channel increases computation cost.
- Converting the image to grayscale was necessary for simpler edge detection.
- Managing correct indexing for pixels and channels was challenging.

---

## 2. How did changing the block size influence the performance of your CUDA implementation?

- Smaller block sizes resulted in lower GPU utilization.
- Larger block sizes improved parallel execution and reduced execution time.
- Very large block sizes increased resource usage and reduced efficiency.
- A block size of `16 × 16` provided balanced performance.

---

## 3. What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.

- Both implementations produced similar edge detection results.
- Minor differences occurred due to floating-point calculations and parallel execution order.
- CUDA implementation executed significantly faster than the CPU version.
- Boundary pixels sometimes differed because of edge handling methods.

---

## 4. Suggest potential optimizations for improving the performance of the Sobel filter.

- Use shared memory to reduce global memory access.
- Store Sobel kernels in constant memory.
- Optimize memory coalescing for faster access.
- Use larger image batches for better GPU utilization.
- Reduce unnecessary synchronization operations.

---

# Deliverables

- Modified CUDA Sobel Edge Detection code.
- Performance comparison between CPU and CUDA implementations.
- Execution time analysis with graphs.
- Answers to experimental questions.

---

# Tools Required

- CUDA Toolkit
- NVIDIA GPU
- OpenCV Library
- C++ Compiler
- Visual Studio / VS Code / Google Colab

