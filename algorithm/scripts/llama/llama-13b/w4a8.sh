CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/llama-13b --eval_ppl \
--epochs 20 --output_dir ./log/llama-13b-w4a8 \
--wbits 4 --abits 8 --lwc --let \
--save_dir ./quant/llama-13b-w4a8 \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande