cmake_minimum_required(VERSION 3.23)
project(imgui
    VERSION 1.0.0
    LANGUAGES C CXX)
file(GLOB FILES *.cpp)
add_library(imgui ${FILES})
set_property(TARGET imgui PROPERTY CXX_STANDARD 17)
target_include_directories(imgui
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        )
