# ABQ-LLM: Inference Acceleration with Arbitrary-Bit Quantization for Large Language Models

ABQ-LLM....


## Contents
- [Install](#install)
- [Model Zoo](#abq-llm-model-zoo)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Install
```
conda create -n abq-llm python=3.10.0 -y
conda activate abq-llm
git clone https://github.com/.../ABQ-LLM.git
cd ABQ-LLM
pip install --upgrade pip 
pip install -r requirements.txt
```


## ABQ-LLM Model Zoo
We provide pre-trained ABQ-LLM model zoo for multiple model families, including LLaMa-1&2, OPT.

You can download the pre-trained OmniQuant parameters you need at [Huggingface](...s).

The detailed support list:
| Models  | Sizes                           | W2A16 | W2A16g128 | W2A16g64 | W3A16 |
| ------- | ------------------------------- | ----- | --------- | -------- | ----- |
| LLaMA   | 7B/13B                  | ✅     | ✅         | ✅        | ✅     |
| LLaMA-2 | 7B/13B                      | ✅     | ✅         | ✅        | ✅     |
| OPT     | 1.3B/2.7B/6.7B/13B | ✅     | ✅         | ✅        | ✅     |

| Models       | Sizes                           | W3A16g128 | W4A16 | W4A16g128 | W6A6 | W4A4 | W4A8 | W3A8 | W2A8 |
| ------------ | ------------------------------- | --------- | ----- | --------- | ---- | ---- |---- |---- |---- |
| LLaMA        | 7B/13B                  | ✅         | ✅     | ✅         | ✅    | ✅    | ✅    | ✅    | ✅    |
| LLaMA-2      | 7B/13B                     | ✅         | ✅     | ✅         | ✅    | ✅    | ✅    | ✅    | ✅    |
| OPT          | 1.3B/2.7B/6.7B/13B | ✅         | ✅     | ✅         | ✅    | ✅    | ✅    | ✅    | ✅    |


## Usage
**We provide full script to run OmniQuant in `./scripts/`**. We use LLaMa-7B as an example here:
1. Obtain the channel-wise scales and shifts required for initialization:
```
conda install git git-lfs
git lfs install
git clone https://huggingface.co/.../act_shifts
git clone https://huggingface.co/.../act_scales
```

Optional, we also offer the script that you can generate channel-wise scales and shifts by yourself:
```
python generate_act_scale_shift.py --model /PATH/TO/LLaMA/llama-7b
```

2. Weight-only quantization
```
# W3A16
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w3a16 \
--eval_ppl --wbits 3 --abits 16  --lwc --let

# W3A16g128
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w3a16g128 \
--eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc --let
```

3. weight-activation quantization
```
# W4A4
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

More detailed and optional arguments:
- `--model`: the local model path or huggingface format.
- `--wbits`: weight quantization bits.
- `--abits`: activation quantization bits.
- `--group_size`: group size of weight quantization. If no set, use per-channel quantization for weight as default.
- `--lwc`: activate the Learnable Weight Clipping (LWC).
- `--let`: activate the Learnable Equivalent Transformation (LET).
- `--lwc_lr`: learning rate of LWC parameters, 1e-2 as default.
- `--let_lr`: learning rate of LET parameters, 5e-3 as default.
- `--epochs`: training epochs. You can set it as 0 to evaluate pre-trained OmniQuant checkpoints.
- `--nsamples`: number of calibration samples, 128 as default.
- `--eval_ppl`: evaluating the perplexity of quantized models.
- `--tasks`: evaluating zero-shot tasks.
- `--resume`: loading pre-trained OmniQuant parameters.
- `--multigpu`: to inference larger network on multiple GPUs
- `--real_quant`: real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.
- `--save_dir`: saving the quantization model for further exploration.




## Results
- ABQ-LLM achieve SoTA performance in weight-only quantization
![weight_only](imgs/...png)
- ABQ-LLM achieve SoTA performance in weight-activation quantization
![weight_activation](imgs/weight_activation.png)


## Related Project
[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[RPTQ: Reorder-Based Post-Training Quantization for Large Language Models](https://github.com/hahnyuan/RPTQ4LLM)

[OmniQuant is a simple and powerful quantization technique for LLMs](https://github.com/OpenGVLab/OmniQuant)


## Citation
If you use our ABQ-LLM approach in your research, please cite our paper:
```
@article{
}
```
