cmake_minimum_required(VERSION 3.24)
project(wsl_cuda_gl
  VERSION 0.1.0
  DESCRIPTION "CUDA <-> OpenGL buffer sharing on WSL2, without host copies"
  HOMEPAGE_URL "https://github.com/tylercasper/wsl-cuda-gl"
  LANGUAGES CXX)

option(WSLCG_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(WSLCG_BUILD_EXAMPLES "Build the GLFW examples" ${PROJECT_IS_TOP_LEVEL})
option(WSLCG_BUILD_TOOLS "Build the probe_wsl_interop diagnostic" ${PROJECT_IS_TOP_LEVEL})
option(WSLCG_INSTALL "Generate install rules" ${PROJECT_IS_TOP_LEVEL})
set(WSLCG_DIRECTX_HEADERS AUTO CACHE STRING
  "Where DirectX-Headers come from: AUTO (system, else download), SYSTEM or FETCH (download)")
set_property(CACHE WSLCG_DIRECTX_HEADERS PROPERTY STRINGS AUTO SYSTEM FETCH)

include(GNUInstallDirs)

# The library itself needs only the CUDA headers (it loads libcuda at runtime) and
# DirectX-Headers (header-only, private). No nvcc, GL headers or link-time GPU libs.
find_package(CUDAToolkit REQUIRED)
include(cmake/DirectXHeaders.cmake)

add_library(wsl_cuda_gl
  src/wsl_cuda_gl.cpp
  src/gl_api.cpp
  src/cuda_api.cpp
  src/d3d12_bridge.cpp
  src/mesa_version.cpp)
add_library(wsl_cuda_gl::wsl_cuda_gl ALIAS wsl_cuda_gl)
target_compile_features(wsl_cuda_gl PUBLIC cxx_std_17)
target_include_directories(wsl_cuda_gl
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    src)
target_include_directories(wsl_cuda_gl SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(wsl_cuda_gl PRIVATE $<BUILD_INTERFACE:wslcg_directx_headers> ${CMAKE_DL_LIBS})
set_target_properties(wsl_cuda_gl PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})

# Tests, examples and the probe compile CUDA kernels; the library does not.
if(WSLCG_BUILD_TESTS OR WSLCG_BUILD_EXAMPLES OR WSLCG_BUILD_TOOLS)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Works without a GPU present (unlike "native"); pass -DCMAKE_CUDA_ARCHITECTURES=native
    # for faster local builds.
    set(CMAKE_CUDA_ARCHITECTURES all-major)
  endif()
  enable_language(CUDA)
  add_subdirectory(support)
endif()
if(WSLCG_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(WSLCG_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if(WSLCG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(WSLCG_INSTALL)
  include(CMakePackageConfigHelpers)
  install(TARGETS wsl_cuda_gl EXPORT wsl_cuda_gl_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES include/wsl_cuda_gl.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  set(config_dir ${CMAKE_INSTALL_LIBDIR}/cmake/wsl_cuda_gl)
  install(EXPORT wsl_cuda_gl_targets NAMESPACE wsl_cuda_gl:: DESTINATION ${config_dir})
  configure_package_config_file(cmake/wsl_cuda_glConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/wsl_cuda_glConfig.cmake INSTALL_DESTINATION ${config_dir})
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/wsl_cuda_glConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/wsl_cuda_glConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/wsl_cuda_glConfigVersion.cmake
    DESTINATION ${config_dir})
  install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/wsl_cuda_gl)
endif()
