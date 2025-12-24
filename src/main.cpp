#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "cpu_pipeline.hpp"
#ifdef ENABLE_CUDA
#include "cuda_kernels.cuh"
#endif

int main(int argc, char** argv) {
    std::string backend = "cpu";
    std::string input = "webcam";
    std::string path = "0";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--input" && i + 1 < argc) input = argv[++i];
        else if (a == "--path" && i + 1 < argc) path = argv[++i];
    }

    CpuParams cpu_p;
    cpu_p.resize_w = 640;
    cpu_p.resize_h = 480;

    cv::VideoCapture cap;
    if (input == "webcam") {
        int idx = std::stoi(path);
        cap.open(idx);
    } else if (input == "video") {
        cap.open(path);
    }

    if (!cap.isOpened()) {
        std::cerr << "Could not open camera/video\n";
        return 1;
    }

    cv::Mat frame, out;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (backend == "cpu") {
            out = run_cpu_pipeline(frame, cpu_p);
        }
#ifdef ENABLE_CUDA
        else if (backend == "cuda") {
            CudaParams cuda_p;
            cuda_p.resize_w = 640;
            cuda_p.resize_h = 480;
            out = run_cuda_pipeline(frame, cuda_p);
        }
#endif
        else {
            std::cerr << "Unknown backend\n";
            return 2;
        }

        cv::imshow("Edge Overlay", out);
        if (cv::waitKey(1) == 27) break; // ESC to quit
    }

    return 0;
}
