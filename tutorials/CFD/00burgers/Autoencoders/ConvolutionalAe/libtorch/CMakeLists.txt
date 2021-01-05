cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(train)

find_package(Torch REQUIRED)

add_executable(train train.cpp)
target_link_libraries(train "${TORCH_LIBRARIES}")
set_property(TARGET train PROPERTY CXX_STANDARD 14)