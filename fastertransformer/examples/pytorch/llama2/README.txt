python3 llama_example.py --tensor_para_size=1 --pipeline_para_size=1 \
                        --ckpt_path /opt/tiger/llama_cc/1-gpu \
                        --tokenizer_path /opt/tiger/llama_cc/tokenizer \
                        --lib_path /opt/tiger/FasterTransformer/build_release/lib/libth_transformer.so \
                        --start_id_file start_ids.csv \
                        --max_batch_size 1 \
                        --time