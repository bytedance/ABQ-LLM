CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/llama-7b --eval_ppl \
--epochs 40 --output_dir ./log/llama-7b-w8a8 \
--wbits 8 --abits 8 --lwc --let \
--save_dir ./quant/llama-7b-w8a8 \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande