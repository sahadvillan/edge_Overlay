#pragma once
#include <cstdint>
#include <opencv2/opencv.hpp>

// CUDA backend params
struct CudaParams {
  int resize_w = 640;
  int resize_h = 480;
  float edge_thresh = 80.0f;
  float overlay_alpha = 0.6f;
};

#ifdef ENABLE_CUDA
cv::Mat run_cuda_pipeline(const cv::Mat& bgr, const CudaParams& p);
#else
inline cv::Mat run_cuda_pipeline(const cv::Mat&, const CudaParams&) {
  throw std::runtime_error("CUDA backend not enabled (rebuild with -DENABLE_CUDA=ON and CUDA Toolkit installed).");
}
#endif
