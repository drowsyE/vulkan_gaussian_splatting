set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.10.0)
project(vulkan_gaussian_splatting VERSION 0.1.0 LANGUAGES C CXX)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_library(vk_radix_sort INTERFACE)

target_include_directories(vk_radix_sort INTERFACE 
    ${CMAKE_SOURCE_DIR}/lib
)

add_executable(gaussian_splatting 
               main.cpp
               src/engine.cpp
               src/engine_utils.cpp
               src/gs_core.cpp
               src/cam.cpp)

target_link_libraries(gaussian_splatting
    Vulkan::Vulkan
    glfw
    glm::glm
    vk_radix_sort
)

target_include_directories(gaussian_splatting PRIVATE 
    ${CMAKE_SOURCE_DIR} 
    ${CMAKE_SOURCE_DIR}/include)