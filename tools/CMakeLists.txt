# Developer diagnostic: which CUDA <-> D3D12 sharing mechanisms work on this machine.
add_executable(probe_wsl_interop probe_wsl_interop.cpp)
target_link_libraries(probe_wsl_interop PRIVATE wslcg_support wslcg_directx_headers CUDA::cuda_driver)
