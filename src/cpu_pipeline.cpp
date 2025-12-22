#include "cpu_pipeline.hpp"
#include <opencv2/imgproc.hpp>

cv::Mat run_cpu_pipeline(const cv::Mat& bgr, const CpuParams& p) {
  cv::Mat frame;
  if (p.resize_w > 0 && p.resize_h > 0) {
    cv::resize(bgr, frame, cv::Size(p.resize_w, p.resize_h), 0, 0, cv::INTER_AREA);
  } else {
    frame = bgr;
  }

  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  cv::Mat blur;
  cv::GaussianBlur(gray, blur, cv::Size(5,5), 1.0);

  cv::Mat gx, gy;
  cv::Sobel(blur, gx, CV_32F, 1, 0, p.sobel_ksize);
  cv::Sobel(blur, gy, CV_32F, 0, 1, p.sobel_ksize);

  cv::Mat mag;
  cv::magnitude(gx, gy, mag);

  cv::Mat edges;
  cv::threshold(mag, edges, p.edge_thresh, 255.0, cv::THRESH_BINARY);
  edges.convertTo(edges, CV_8U);

  // make edges red overlay
  cv::Mat overlay = frame.clone();
  for (int y=0; y<edges.rows; ++y) {
    const uint8_t* e = edges.ptr<uint8_t>(y);
    cv::Vec3b* o = overlay.ptr<cv::Vec3b>(y);
    for (int x=0; x<edges.cols; ++x) {
      if (e[x]) {
        // BGR: red is (0,0,255)
        o[x] = cv::Vec3b(0, 0, 255);
      }
    }
  }

  cv::Mat out;
  cv::addWeighted(frame, 1.0 - p.overlay_alpha, overlay, p.overlay_alpha, 0.0, out);
  return out;
}
