# FasterTransformer(ABQ Version)

[FasterTransformer](https://github.com/NVIDIA/FasterTransformer) provides a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA. LLama model was integrated in [void-main/FasterTransformer](https://github.com/void-main/FasterTransformer). To evaluate end-to-end latency, this codebase is modified from [void-main/FasterTransformer](https://github.com/void-main/FasterTransformer).

Note that current codebase is for efficiency evaluation. We use random weights therefore no meaningful output.

## Install

1. Compile the project
```
bash build.sh
``` 

## Usage
1. Config llama
Change precision in examples/cpp/llama/llama_config.ini
```
fp16:  int8_mode=0
w8a16: int8_mode=1
w8a8:  int8_mode=2
w4a16: int8_mode=4
w2a8:  int8_mode=5
```

2. Run llama on single GPU
```
cd build_release
./bin/llama_example
```

3. (Optional )Run in multi GPU
Change tensor_para_size=2 in examples/cpp/llama/llama_config.ini

```
cd build_release
mpirun -n 2 ./bin/llama_example
```