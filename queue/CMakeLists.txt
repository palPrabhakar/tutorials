cmake_minimum_required(VERSION 3.10)

project(queue C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_C_FLAGS "-Wall -Wextra -g -O2")

add_subdirectory(test)
# add_subdirectory(src)


