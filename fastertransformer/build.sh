TYPE=$1

SPARSE_PATH=$MY_ENV/../libcusparse_lt

if [ "$TYPE" = 'Debug' ]; then
    echo 'Build Debug Version'
    if [ ! -d build_debug ]; then
        mkdir build_debug
    fi
    cd build_debug

    cmake .. \
        -DSM=80 \
        -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_MULTI_GPU=ON \
        -DBUILD_CUTLASS_MIXED_GEMM=ON \
        -DBUILD_CUTLASS_MOE=OFF \
        -DBUILD_PYT=OFF \
        -DUSE_NVTX=ON  \
        -DSPARSITY_SUPPORT=OFF   \
        -DCUSPARSELT_PATH=$SPARSE_PATH \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=on
    make -j128
    
else
    echo 'Build Release Version'
    if [ ! -d build_release ]; then
        mkdir build_release
    fi
    cd build_release

    cmake .. \
        -DSM=80 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_MULTI_GPU=ON \
        -DBUILD_CUTLASS_MIXED_GEMM=ON \
        -DBUILD_CUTLASS_MOE=OFF \
        -DBUILD_PYT=OFF \
        -DUSE_NVTX=ON \
        -DSPARSITY_SUPPORT=OFF   \
        -DCUSPARSELT_PATH=$SPARSE_PATH \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    make -j128
fi
