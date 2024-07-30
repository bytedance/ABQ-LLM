CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/llama-7b --eval_ppl \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--wbits 4 --abits 4 --lwc --let \
--save_dir ./quant/llama-7b-w4a4 \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande