CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log/Llama-2-7b-w6a6 \
--wbits 6 --abits 6 --lwc --let \
--save_dir ./quant/Llama-2-7b-a6a6 \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande