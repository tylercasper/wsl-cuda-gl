# GLFW demos. Each takes --help; see README.md.
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(glfw3 3.3 REQUIRED)

add_library(wslcg_example_app STATIC app.cpp)
target_include_directories(wslcg_example_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wslcg_example_app PUBLIC wsl_cuda_gl::wsl_cuda_gl wslcg_support glfw OpenGL::OpenGL)

foreach(example host_copy zero_copy pipelined)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE wslcg_example_app)
endforeach()
