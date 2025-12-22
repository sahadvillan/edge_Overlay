#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

#include "args.hpp"
#include "cpu_pipeline.hpp"
#include "cuda_kernels.cuh"

static cv::Mat make_synthetic(int w, int h) {
  cv::Mat img(h, w, CV_8UC3, cv::Scalar(20, 20, 20));
  cv::rectangle(img, {w/8, h/6}, {w*7/8, h*5/6}, cv::Scalar(40, 220, 40), -1);
  cv::circle(img, {w/2, h/2}, std::min(w,h)/5, cv::Scalar(220, 40, 40), -1);
  cv::putText(img, "Synthetic Frame", {20, h-30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(240,240,240), 2);
  // add some noise
  cv::Mat noise(h, w, CV_8SC3);
  cv::randn(noise, 0, 20);
  cv::Mat tmp;
  img.convertTo(tmp, CV_16SC3);
  tmp += noise;
  tmp.convertTo(img, CV_8UC3);
  return img;
}

static void print_usage() {
  std::cout <<
R"(Usage:
  app --backend cpu|cuda [--input webcam|video|image] [--path <path>] [--frames N] [--save_dir outputs]

Examples:
  app --backend cpu  --input webcam --path 0
  app --backend cuda --input video  --path input.mp4
  app --backend cpu  --input image  --path samples/lena.png --save_dir outputs
  app --backend cpu  --input none   (runs synthetic)

Notes:
  - For webcam, pass --path 0 (or 1,2,...)
  - If CUDA not available, build with -DENABLE_CUDA=OFF or run --backend cpu
)";
}

int main(int argc, char** argv) {
  auto args = parse_args(argc, argv);
  if (args.has("help") || args.has("h")) { print_usage(); return 0; }

  std::string backend = args.get("backend", "cpu");
  std::string input   = args.get("input", "none"); // webcam|video|image|none
  std::string path    = args.get("path", "");
  int frames_limit    = args.get_int("frames", 300);
  std::string save_dir = args.get("save_dir", "");
  bool show = args.get_bool("show", true);

  if (!save_dir.empty()) {
    std::filesystem::create_directories(save_dir);
  }

  CpuParams cpu_p;
  cpu_p.resize_w = args.get_int("w", 640);
  cpu_p.resize_h = args.get_int("hgt", 480);
  cpu_p.edge_thresh = std::stod(args.get("edge_thresh", "80"));
  cpu_p.overlay_alpha = std::stod(args.get("alpha", "0.6"));

  CudaParams cuda_p;
  cuda_p.resize_w = cpu_p.resize_w;
  cuda_p.resize_h = cpu_p.resize_h;
  cuda_p.edge_thresh = (float)cpu_p.edge_thresh;
  cuda_p.overlay_alpha = (float)cpu_p.overlay_alpha;

  cv::VideoCapture cap;
  cv::Mat single_image;

  if (input == "webcam") {
    int idx = 0;
    try { idx = std::stoi(path.empty() ? "0" : path); } catch(...) { idx = 0; }
    cap.open(idx);
    if (!cap.isOpened()) {
      std::cerr << "Failed to open webcam index " << idx << "\n";
      return 1;
    }
  } else if (input == "video") {
    cap.open(path);
    if (!cap.isOpened()) {
      std::cerr << "Failed to open video: " << path << "\n";
      return 1;
    }
  } else if (input == "image") {
    single_image = cv::imread(path, cv::IMREAD_COLOR);
    if (single_image.empty()) {
      std::cerr << "Failed to read image: " << path << "\n";
      return 1;
    }
  } else if (input == "none") {
    // synthetic
  } else {
    std::cerr << "Unknown --input: " << input << "\n";
    print_usage();
    return 1;
  }

  std::vector<double> ms;
  ms.reserve(frames_limit);

  auto process_one = [&](const cv::Mat& frame)->cv::Mat {
    if (backend == "cpu") {
      return run_cpu_pipeline(frame, cpu_p);
    } else if (backend == "cuda") {
      return run_cuda_pipeline(frame, cuda_p);
    } else {
      throw std::runtime_error("Unknown backend: " + backend);
    }
  };

  int frame_id = 0;
  while (true) {
    cv::Mat frame;
    if (input == "image") {
      frame = single_image.clone();
    } else if (input == "none") {
      frame = make_synthetic(cpu_p.resize_w > 0 ? cpu_p.resize_w : 640,
                             cpu_p.resize_h > 0 ? cpu_p.resize_h : 480);
    } else {
      cap >> frame;
      if (frame.empty()) break;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat out;
    try {
      out = process_one(frame);
    } catch (const std::exception& e) {
      std::cerr << "Processing error: " << e.what() << "\n";
      return 2;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms.push_back(dt);

    if (show) {
      cv::imshow("out", out);
      int key = cv::waitKey(input == "image" || input == "none" ? 0 : 1);
      if (key == 27 || key == 'q') break;
    }

    if (!save_dir.empty()) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "frame_%06d.png", frame_id);
      cv::imwrite((std::filesystem::path(save_dir) / buf).string(), out);
    }

    frame_id++;
    if (input == "image" || input == "none") break; // single run by default
    if (frame_id >= frames_limit) break;
  }

  if (!ms.empty()) {
    double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / (double)ms.size();
    auto ms_sorted = ms;
    std::sort(ms_sorted.begin(), ms_sorted.end());
    auto p95 = ms_sorted[(size_t)(0.95 * (ms_sorted.size()-1))];

    double fps = (mean > 0.0) ? (1000.0 / mean) : 0.0;

    std::cout << "Backend: " << backend << "\n";
    std::cout << "Frames: " << ms.size() << "\n";
    std::cout << "Mean latency (ms): " << mean << "\n";
    std::cout << "P95 latency (ms): " << p95 << "\n";
    std::cout << "Approx FPS: " << fps << "\n";
  }

  return 0;
}
