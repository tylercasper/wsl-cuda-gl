# Provides the INTERFACE target `wslcg_directx_headers`: the include directories of
# Microsoft's DirectX-Headers (https://github.com/microsoft/DirectX-Headers), which the
# library needs at build time only. Nothing is linked: the D3D12 runtime is loaded from
# the WSL driver at run time, and the GUIDs used are defined locally.

add_library(wslcg_directx_headers INTERFACE)

set(_wslcg_dxh_found FALSE)
if(NOT WSLCG_DIRECTX_HEADERS STREQUAL "FETCH")
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(WSLCG_DXH QUIET DirectX-Headers>=1.600)
  endif()
  if(WSLCG_DXH_FOUND)
    target_include_directories(wslcg_directx_headers SYSTEM INTERFACE ${WSLCG_DXH_INCLUDE_DIRS})
    set(_wslcg_dxh_found TRUE)
    message(STATUS "wsl_cuda_gl: using system DirectX-Headers ${WSLCG_DXH_VERSION}")
  elseif(WSLCG_DIRECTX_HEADERS STREQUAL "SYSTEM")
    message(FATAL_ERROR "wsl_cuda_gl: DirectX-Headers not found via pkg-config "
                        "(Debian/Ubuntu: apt install directx-headers-dev), or use -DWSLCG_DIRECTX_HEADERS=FETCH")
  endif()
endif()

if(NOT _wslcg_dxh_found)
  include(FetchContent)
  FetchContent_Declare(wslcg_dxh
    URL https://github.com/microsoft/DirectX-Headers/archive/refs/tags/v1.614.1.tar.gz
    URL_HASH SHA256=344eb454c979ea68d8255d82c818bf7daf01f5109d26ac104f9911d18fae3b21
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # Headers only: a SOURCE_SUBDIR without a CMakeLists.txt makes MakeAvailable
    # download without adding the project.
    SOURCE_SUBDIR headers-only)
  FetchContent_MakeAvailable(wslcg_dxh)
  target_include_directories(wslcg_directx_headers SYSTEM INTERFACE
    ${wslcg_dxh_SOURCE_DIR}/include
    ${wslcg_dxh_SOURCE_DIR}/include/directx
    ${wslcg_dxh_SOURCE_DIR}/include/wsl/stubs)
  message(STATUS "wsl_cuda_gl: using downloaded DirectX-Headers 1.614.1")
endif()
