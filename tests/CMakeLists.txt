# Test suites (run with ctest):
#
#   unit/      no GPU needed; run everywhere, including CI
#   gpu/       need WSL2 + an NVIDIA GPU + Mesa's d3d12 driver; skip themselves
#              (exit 77) elsewhere. WSLCG_REQUIRE_GPU_TESTS=1 makes a skip a failure.
#   consumer/  build a separate project against the library, as users would
#
# Labels: `ctest -L unit`, `ctest -L gpu`, `ctest -L consumer`.

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(glfw3 3.3 REQUIRED) # only for the header-compatibility test

add_library(wslcg_test_support STATIC support/gl_env.cpp)
target_include_directories(wslcg_test_support PUBLIC support)
target_link_libraries(wslcg_test_support PUBLIC wsl_cuda_gl::wsl_cuda_gl wslcg_support OpenGL::OpenGL OpenGL::EGL)

# Tests of internals include the library's private headers; the symbols are visible in
# both static and shared builds.
set(internal_includes ${PROJECT_SOURCE_DIR}/src ${CUDAToolkit_INCLUDE_DIRS})

function(wslcg_test name label)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE wslcg_test_support)
  target_include_directories(${name} PRIVATE ${internal_includes})
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS ${label} SKIP_RETURN_CODE 77 TIMEOUT 300)
endfunction()

# --- unit ---------------------------------------------------------------------
wslcg_test(test_mesa_version unit unit/test_mesa_version.cpp)
wslcg_test(test_vtable_slot unit unit/test_vtable_slot.cpp unit/vtable_slot_c_layout.cpp)
target_link_libraries(test_vtable_slot PRIVATE wslcg_directx_headers)
wslcg_test(test_no_gl_context unit unit/test_no_gl_context.cpp)
wslcg_test(test_wrong_gl_driver unit unit/test_wrong_gl_driver.cpp)
wslcg_test(test_public_header unit unit/test_public_header.cpp unit/public_header_after_gl_cuda.cpp)
target_link_libraries(test_public_header PRIVATE glfw)
wslcg_test(test_public_header_cxx20 unit unit/test_public_header.cpp unit/public_header_after_gl_cuda.cpp)
target_link_libraries(test_public_header_cxx20 PRIVATE glfw)
set_target_properties(test_public_header_cxx20 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# --- gpu ----------------------------------------------------------------------
foreach(t cuda_to_gl gl_to_cuda copy_helpers pipelined sync map_contract lifecycle support_report mesa_workaround)
  wslcg_test(test_${t} gpu gpu/test_${t}.cpp)
endforeach()
if(TARGET probe_wsl_interop)
  # The probe links libcuda, so it can't even start without the NVIDIA driver: check
  # for WSL's GPU device first.
  add_test(NAME probe_gpu_address
    COMMAND sh -c "test -e /dev/dxg || { echo 'SKIPPED: no /dev/dxg'; exit 77; }; exec \"$0\" gpuva"
            $<TARGET_FILE:probe_wsl_interop>)
  set_tests_properties(probe_gpu_address PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endif()

# --- consumer -----------------------------------------------------------------
if(WSLCG_INSTALL)
  set(prefix ${CMAKE_CURRENT_BINARY_DIR}/consumer-prefix)
  add_test(NAME consumer_install COMMAND ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR} --prefix ${prefix})
  set_tests_properties(consumer_install PROPERTIES LABELS consumer FIXTURES_SETUP wslcg_installed)
endif()
foreach(mode find_package add_subdirectory fetchcontent)
  if(mode STREQUAL "find_package" AND NOT WSLCG_INSTALL)
    continue()
  endif()
  add_test(NAME consumer_${mode}
    COMMAND ${CMAKE_CTEST_COMMAND}
      --build-and-test ${CMAKE_CURRENT_SOURCE_DIR}/consumer ${CMAKE_CURRENT_BINARY_DIR}/consumer-${mode}
      --build-generator ${CMAKE_GENERATOR}
      --build-options
        -DWSLCG_CONSUMER_MODE=${mode}
        -DWSLCG_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/consumer-prefix
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCUDAToolkit_ROOT=${CUDAToolkit_LIBRARY_ROOT}
        -DWSLCG_DIRECTX_HEADERS=${WSLCG_DIRECTX_HEADERS}
        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
      --test-command consumer)
  set_tests_properties(consumer_${mode} PROPERTIES LABELS consumer TIMEOUT 600)
  if(mode STREQUAL "find_package")
    set_tests_properties(consumer_${mode} PROPERTIES FIXTURES_REQUIRED wslcg_installed)
  endif()
endforeach()
