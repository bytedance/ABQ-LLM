
TYPE=$1

# ./bin/llamaV2_example
# ./bin/llamaV2_example

CMD=$2     

nsys_func(){
    nsys profile \
    --trace=cublas,cuda,cudnn,nvtx \
    --cuda-memory-usage=true    \
    --output=./profling   \
    --export=sqlite \
    --stats=true    \
    --show-output=true  \
    --force-overwrite=true \
    ${CMD}
}

if [ "$TYPE" = 'Debug' ]; then
    echo 'Profling Debug Version ${CMD}'
    if [ ! -d build_debug ]; then
        bash build.sh Debug
    fi
    cd build_debug
    nsys_func

else
    echo 'Profling Release Version ${CMD}'
    if [ ! -d build_release ]; then
        bash build.sh Release
    fi
    cd build_release
    nsys_func
fi
