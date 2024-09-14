
cmake_minimum_required(VERSION 3.8)
project(abqgemm)

if(FIND_SM_80 STREQUAL True)
    add_definitions(-DGPU_ARCH=80)
else()
    if(FIND_SM_75 STREQUAL True)
        add_definitions(-DGPU_ARCH=75)
    endif()
endif()

file(GLOB ABQ_FILES ${ABQ_FILES} ./common/*.cu)
file(GLOB ABQ_FILES ${ABQ_FILES} ./*.cu)
file(GLOB ABQ_FILES ${ABQ_FILES} ./mma_any/*.cu)
file(GLOB ABQ_FILES ${ABQ_FILES} ./mma_any/aq_wmma_impl/*.cu)
file(GLOB ABQ_FILES ${ABQ_FILES} ./mma_any/aq_bmma_impl/*.cu)

add_library(abqgemm STATIC abq_gemm_wrapper.cu ${ABQ_FILES})
set_property(TARGET abqgemm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET abqgemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

