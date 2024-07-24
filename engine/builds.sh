
CUTLASS_PATH = "../../../3rdparty/cuda/cutlass"

    INCLUDE_FLAG = "-I$CUTLASS_PATH/tools/util/include \
             -I$CUTLASS_PATH/include \
             -I./ "

    BUILD_FLAG = " -std=c++17 -g -w -arch=sm_86 "

                 nvcc $INCLUDE_FLAG $BUILD_FLAG -
                 o test_w4a4_int_base.exe test_w4a4_int_base.cu
