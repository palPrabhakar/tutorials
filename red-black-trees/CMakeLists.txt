cmake_minimum_required(VERSION 3.10)

project(Red-Black-Trees C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_C_FLAGS "-Wall -Wextra -g")


add_subdirectory(tests)
add_subdirectory(src)


