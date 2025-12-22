#include "cuda_kernels.cuh"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
  }
}

// BGR (uchar3) -> Gray (uint8)
__global__ void bgr_to_gray(const uchar3* bgr, uint8_t* gray, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  int idx = y * w + x;
  uchar3 p = bgr[idx];
  // ITU-R BT.601 luma approx
  float g = 0.114f * p.x + 0.587f * p.y + 0.299f * p.z; // p.x=B, p.y=G, p.z=R
  gray[idx] = (uint8_t)(g + 0.5f);
}

// 3x3 box blur (simple + fast enough demo)
__global__ void box_blur_3x3(const uint8_t* in, uint8_t* out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  int sum = 0;
  int cnt = 0;
  for (int dy=-1; dy<=1; ++dy) {
    int yy = y + dy;
    if (yy < 0 || yy >= h) continue;
    for (int dx=-1; dx<=1; ++dx) {
      int xx = x + dx;
      if (xx < 0 || xx >= w) continue;
      sum += in[yy*w + xx];
      cnt++;
    }
  }
  out[y*w + x] = (uint8_t)(sum / cnt);
}

// Sobel magnitude -> edges (0/255)
__global__ void sobel_edges(const uint8_t* in, uint8_t* edges, int w, int h, float thresh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  if (x == 0 || y == 0 || x == w-1 || y == h-1) {
    edges[y*w + x] = 0;
    return;
  }

  int idx = y*w + x;

  int tl = in[(y-1)*w + (x-1)];
  int tc = in[(y-1)*w + (x)];
  int tr = in[(y-1)*w + (x+1)];
  int ml = in[(y)*w + (x-1)];
  int mr = in[(y)*w + (x+1)];
  int bl = in[(y+1)*w + (x-1)];
  int bc = in[(y+1)*w + (x)];
  int br = in[(y+1)*w + (x+1)];

  int gx = -tl + tr - 2*ml + 2*mr - bl + br;
  int gy = -tl - 2*tc - tr + bl + 2*bc + br;

  float mag = sqrtf((float)(gx*gx + gy*gy));
  edges[idx] = (mag >= thresh) ? 255 : 0;
}

// Overlay edges as red onto BGR
__global__ void overlay_edges_red(const uchar3* bgr_in, const uint8_t* edges,
                                  uchar3* bgr_out, int w, int h, float alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  int idx = y*w + x;
  uchar3 p = bgr_in[idx];
  uint8_t e = edges[idx];

  if (!e) {
    bgr_out[idx] = p;
    return;
  }

  // Blend with red (0,0,255) in BGR
  float b = (1.0f - alpha) * p.x + alpha * 0.0f;
  float g = (1.0f - alpha) * p.y + alpha * 0.0f;
  float r = (1.0f - alpha) * p.z + alpha * 255.0f;

  uchar3 o;
  o.x = (uint8_t)(b + 0.5f);
  o.y = (uint8_t)(g + 0.5f);
  o.z = (uint8_t)(r + 0.5f);
  bgr_out[idx] = o;
}

cv::Mat run_cuda_pipeline(const cv::Mat& bgr_in, const CudaParams& p) {
  cv::Mat frame;
  if (p.resize_w > 0 && p.resize_h > 0) {
    cv::resize(bgr_in, frame, cv::Size(p.resize_w, p.resize_h), 0, 0, cv::INTER_AREA);
  } else {
    frame = bgr_in;
  }

  if (frame.empty() || frame.type() != CV_8UC3) {
    throw std::runtime_error("Input frame must be CV_8UC3 (BGR).");
  }

  int w = frame.cols;
  int h = frame.rows;
  size_t npx = (size_t)w * (size_t)h;

  uchar3* d_bgr_in = nullptr;
  uchar3* d_bgr_out = nullptr;
  uint8_t* d_gray = nullptr;
  uint8_t* d_blur = nullptr;
  uint8_t* d_edges = nullptr;

  ck(cudaMalloc(&d_bgr_in,  npx * sizeof(uchar3)), "cudaMalloc d_bgr_in");
  ck(cudaMalloc(&d_bgr_out, npx * sizeof(uchar3)), "cudaMalloc d_bgr_out");
  ck(cudaMalloc(&d_gray,    npx * sizeof(uint8_t)), "cudaMalloc d_gray");
  ck(cudaMalloc(&d_blur,    npx * sizeof(uint8_t)), "cudaMalloc d_blur");
  ck(cudaMalloc(&d_edges,   npx * sizeof(uint8_t)), "cudaMalloc d_edges");

  ck(cudaMemcpy(d_bgr_in, frame.ptr(), npx * sizeof(uchar3), cudaMemcpyHostToDevice),
     "cudaMemcpy H2D bgr");

  dim3 block(16,16);
  dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);

  bgr_to_gray<<<grid, block>>>(d_bgr_in, d_gray, w, h);
  box_blur_3x3<<<grid, block>>>(d_gray, d_blur, w, h);
  sobel_edges<<<grid, block>>>(d_blur, d_edges, w, h, p.edge_thresh);
  overlay_edges_red<<<grid, block>>>(d_bgr_in, d_edges, d_bgr_out, w, h, p.overlay_alpha);

  ck(cudaGetLastError(), "CUDA kernel launch");
  ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  cv::Mat out(h, w, CV_8UC3);
  ck(cudaMemcpy(out.ptr(), d_bgr_out, npx * sizeof(uchar3), cudaMemcpyDeviceToHost),
     "cudaMemcpy D2H out");

  cudaFree(d_bgr_in);
  cudaFree(d_bgr_out);
  cudaFree(d_gray);
  cudaFree(d_blur);
  cudaFree(d_edges);

  return out;
}
#endif
