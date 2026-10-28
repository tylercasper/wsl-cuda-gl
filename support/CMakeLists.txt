# CUDA kernels shared by the examples, tests and probe (a deterministic test image, a
# simulated workload, a plain copy). Not part of the library.
add_library(wslcg_support STATIC pattern.cu)
target_include_directories(wslcg_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Static cudart: test binaries must start (and skip) on machines without CUDA installed.
target_link_libraries(wslcg_support PUBLIC CUDA::cudart_static)
target_compile_options(wslcg_support PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
