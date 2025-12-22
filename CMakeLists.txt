cmake_minimum_required(VERSION 3.20)
project(realtime_edge_overlay LANGUAGES CXX)

option(ENABLE_CUDA "Enable CUDA backend" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(app
  src/main.cpp
  src/cpu_pipeline.cpp
)

target_include_directories(app PRIVATE include)
target_link_libraries(app PRIVATE ${OpenCV_LIBS})

if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(app PRIVATE src/cuda_kernels.cu)
  target_compile_definitions(app PRIVATE ENABLE_CUDA=1)

  set_target_properties(app PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )

  # Reasonable defaults
  target_compile_options(app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )

  target_link_libraries(app PRIVATE CUDA::cudart)
endif()
