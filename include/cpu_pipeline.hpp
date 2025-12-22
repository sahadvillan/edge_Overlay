#pragma once
#include <opencv2/opencv.hpp>

struct CpuParams {
  int resize_w = 640;
  int resize_h = 480;
  int sobel_ksize = 3;
  double edge_thresh = 80.0; // magnitude threshold
  double overlay_alpha = 0.6; // edge overlay strength
};

cv::Mat run_cpu_pipeline(const cv::Mat& bgr, const CpuParams& p);
