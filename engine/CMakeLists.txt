cmake_minimum_required(VERSION 3.10)

project(ABQ-LLM LANGUAGES CXX)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/${PLATFORM_ABI})

set(CXX_STD "14" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# setting comman compiler flags
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}  -g -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
if (MSVC)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -w")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -w")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
endif()

## set common include directories 
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA.cmake)

if (CUDA_VERSION VERSION_LESS 11.0)
    add_definitions("-DENABLE_BF16")
    message("CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.0, enable -DENABLE_BF16 flag")
endif()
option(ENABLE_FP8 "ENABLE_FP8" OFF)

option(ENABLE_W2A2 "ENABLE_W2A2" OFF)
option(ENABLE_W2A4 "ENABLE_W2A4" OFF)
option(ENABLE_W2A6 "ENABLE_W2A6" OFF)
option(ENABLE_W2A8 "ENABLE_W2A8" OFF)
option(ENABLE_W3A3 "ENABLE_W3A3" OFF)
option(ENABLE_W4A4 "ENABLE_W4A4" OFF)
option(ENABLE_W4A8 "ENABLE_W4A8" OFF)
option(ENABLE_W5A5 "ENABLE_W5A5" OFF)
option(ENABLE_W6A6 "ENABLE_W6A6" OFF)
option(ENABLE_W7A7 "ENABLE_W7A7" OFF)
option(ENABLE_W8A8 "ENABLE_W8A8" OFF)

if(ENABLE_W2A2)
    add_definitions("-DW2A2")
endif()
if(ENABLE_W2A4)
    add_definitions("-DW2A4")
endif()
if(ENABLE_W2A6)
    add_definitions("-DW2A6")
endif()
if(ENABLE_W2A8)
    add_definitions("-DW2A8")
endif()
if(ENABLE_W3A3)
    add_definitions("-DW3A3")
endif()
if(ENABLE_W4A4)
    add_definitions("-DW4A4")
endif()
if(ENABLE_W4A8)
    add_definitions("-DW4A8")
endif()
if(ENABLE_W5A5)
    add_definitions("-DW5A5")
endif()
if(ENABLE_W6A6)
    add_definitions("-DW6A6")
endif()
if(ENABLE_W7A7)
    add_definitions("-DW7A7")
endif()
if(ENABLE_W8A8)
    add_definitions("-DW8A8")
endif()


set(CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/cutlass/include)
list(APPEND CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/cutlass/tools/util/include)


# setting nvcc compiler flags
set(SM_SETS 75 80 86 89 90)
set(USING_WMMA False)
set(FIND_SM False)
set(FIND_SM_80 False)
set(FIND_SM_75 False)
foreach(SM_NUM IN LISTS SM_SETS)
    string(FIND "${SM}" "${SM_NUM}" SM_POS)
    if(SM_POS GREATER -1)
        set(FIND_SM True)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")
        if (SM_NUM STREQUAL 75 OR SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86 OR SM_NUM STREQUAL 89 OR SM_NUM STREQUAL 90)
            set(USING_WMMA True)        
        endif()
        if (SM_NUM STREQUAL 75)
            set(FIND_SM_75 True)        
        endif()
        if (SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86 OR SM_NUM STREQUAL 89 OR SM_NUM STREQUAL 90)
            set(FIND_SM_80 True)        
        endif()
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
        message("-- Assign GPU architecture (sm=${SM_NUM})")
    endif()
endforeach()
if(USING_WMMA STREQUAL True)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
    message("-- Use WMMA")
endif()

if(FIND_SM_80 STREQUAL True)
    add_definitions(-DGPU_ARCH=80)
else()
    if(FIND_SM_75 STREQUAL True)
        add_definitions(-DGPU_ARCH=75)
    endif()
endif()

if(NOT (FIND_SM STREQUAL True))
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        -gencode=arch=compute_89,code=\\\"sm_86,compute_89\\\" \
                        ")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    message("-- Assign GPU architecture (sm=75,80,86,89)")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}") # -DCUDA_PTX_FP8_F2FP_ENABLED
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O2")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler -O0 -G")
if(BUILD_FAST_MATH)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
endif()
# print compile flgs
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message("CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
message("CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")

# cuda backend header dirs and libs append
list(APPEND COMMON_HEADER_DIRS ${CUDA_PATH}/include)
list(APPEND COMMON_HEADER_DIRS ${CUTLASS_HEADER_DIR})
list(APPEND COMMON_LIB_DIRS ${CUDA_PATH}/lib/x64)

# Add header files and library dependencies
include_directories( SYSTEM
  ${COMMON_HEADER_DIRS}
)
link_directories(
  ${COMMON_LIB_DIRS}
)

set(QUANT_FILES "" CACHE INTERNAL "arbitrarily quant files")
# gemm gemv quant sparse
file(GLOB QUANT_FILES ${QUANT_FILES} ./common/*.cu)
file(GLOB QUANT_FILES ${QUANT_FILES} ./wmma_any/*.cu)
file(GLOB QUANT_FILES ${QUANT_FILES} ./mma_any/*.cu)
file(GLOB QUANT_FILES ${QUANT_FILES} ./mma_any/aq_wmma_impl/*.cu)
add_library(arbitrarily_quant STATIC ${QUANT_FILES})
set_property(TARGET arbitrarily_quant PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET arbitrarily_quant PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(arbitrarily_quant PUBLIC -lcudart)

add_executable(test_w4a4_int_wmma test_w4a4_int_wmma.cu)
target_link_libraries(test_w4a4_int_wmma arbitrarily_quant)
add_executable(test_w4a4_uint_wmma test_w4a4_uint_wmma.cu)
target_link_libraries(test_w4a4_uint_wmma arbitrarily_quant)
add_executable(test_any_wmma test_any_wmma.cu)
target_link_libraries(test_any_wmma arbitrarily_quant)
add_executable(test_any_mma test_any_mma.cu)
target_link_libraries(test_any_mma arbitrarily_quant)
add_executable(test_layout test_layout.cc)
add_executable(gen_kernel gen_kernel.cpp)