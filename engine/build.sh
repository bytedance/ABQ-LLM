



mkdir build_linux
cd build_linux

cmake .. \
        -DSM=86 \
        -DENABLE_W2A2=ON \
        -DENABLE_W2A4=ON \
        -DENABLE_W2A6=ON \
        -DENABLE_W2A8=ON \
        -DENABLE_W3A3=ON \
        -DENABLE_W4A4=ON \
        -DENABLE_W4A8=ON \
        -DENABLE_W5A5=ON \
        -DENABLE_W6A6=ON \
        -DENABLE_W7A7=ON \
        -DENABLE_W8A8=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=on

make -j32
