cmake_minimum_required(VERSION 3.16)
project(matrix_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(matrix_engine INTERFACE)
target_include_directories(matrix_engine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(add_benchmark src/add.cpp)
set_target_properties(add_benchmark PROPERTIES OUTPUT_NAME matrix_add_benchmark)
target_link_libraries(add_benchmark PRIVATE matrix_engine)

add_executable(multiply_benchmark src/multiply.cpp)
set_target_properties(multiply_benchmark PROPERTIES OUTPUT_NAME matrix_benchmark)
target_link_libraries(multiply_benchmark PRIVATE matrix_engine)

add_executable(determinant_benchmark src/determinant.cpp)
set_target_properties(determinant_benchmark PROPERTIES OUTPUT_NAME matrix_determinant_benchmark)
target_link_libraries(determinant_benchmark PRIVATE matrix_engine)

add_executable(matrix_tests src/tests.cpp)
target_link_libraries(matrix_tests PRIVATE matrix_engine)

if(MSVC)
  target_compile_options(add_benchmark PRIVATE /O2)
  target_compile_options(multiply_benchmark PRIVATE /O2)
  target_compile_options(determinant_benchmark PRIVATE /O2)
  target_compile_options(matrix_tests PRIVATE /O2)
else()
  target_compile_options(add_benchmark PRIVATE -O3 -march=native)
  target_compile_options(multiply_benchmark PRIVATE -O3 -march=native)
  target_compile_options(determinant_benchmark PRIVATE -O3 -march=native)
  target_compile_options(matrix_tests PRIVATE -O3)
endif()
