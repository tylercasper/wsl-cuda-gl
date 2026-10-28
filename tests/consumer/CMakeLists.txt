# A downstream project, built by the consumer_* tests the three ways users can pull in
# the library. It is C++-only (no CUDA language) and uses C++20 while the library is
# built as C++17.
cmake_minimum_required(VERSION 3.24)
project(wslcg_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

if(WSLCG_CONSUMER_MODE STREQUAL "find_package")
  find_package(wsl_cuda_gl 0.1 CONFIG REQUIRED)
elseif(WSLCG_CONSUMER_MODE STREQUAL "add_subdirectory")
  add_subdirectory(${WSLCG_SOURCE_DIR} wsl_cuda_gl)
elseif(WSLCG_CONSUMER_MODE STREQUAL "fetchcontent")
  include(FetchContent)
  FetchContent_Declare(wsl_cuda_gl SOURCE_DIR ${WSLCG_SOURCE_DIR})
  FetchContent_MakeAvailable(wsl_cuda_gl)
else()
  message(FATAL_ERROR "set WSLCG_CONSUMER_MODE to find_package, add_subdirectory or fetchcontent")
endif()

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE wsl_cuda_gl::wsl_cuda_gl)
