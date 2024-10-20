cmake_minimum_required(VERSION 3.10)

project(Red-Black-Trees C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

set(SOURCES
    src/main.c
    src/red_black_trees.c
)

# include_directories(include)

add_executable(rbt ${SOURCES})

