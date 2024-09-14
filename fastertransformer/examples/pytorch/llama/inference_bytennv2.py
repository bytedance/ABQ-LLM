
from torch.nn.utils.rnn import pad_sequence
import random
import os
import argparse
import configparser
import torch
import numpy as np
import random
from transformers import  AutoTokenizer
import torch 
import torch.backends.cudnn as cudnn
import json
import timeit
from llamav1 import LlamaV1


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=0.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='/opt/tiger/llama_cc/1-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='/opt/tiger/FasterTransformer/build_release/lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tokenizer_dir', type=str, default="/opt/tiger/llama_cc/tokenizer",
                        help='vocabulary file.')
    parser.add_argument('--start_id', type=int, default=1,
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=2,
                        help='end token id.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--min_length', type=int, default=0,
                        help='A minimum number of tokens to generate')
    parser.add_argument('--max_seq_len', type=int, default=2048,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--total_seq_len', type=int, default=1024,
                        help='max sequence length for max_input_penghth + request output len.')
    parser.add_argument('--inference_data_type', '--data_type', type=str, choices=['fp32', 'fp16', 'bf16', 'fp8'], default='fp16')
    parser.add_argument('--weights_data_type', type=str, default="fp16", choices=["fp32", "fp16"],  help='Data type of ByteNN checkpoint weights',)
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--sample_input_file', type=str, default="/opt/tiger/transformer-story-bot/tests/t2d_tests/input_prompts.json",
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file', type=str, default="/opt/tiger/transformer-story-bot/tests/t2d_tests/bytenn_prompt_output.txt",
                        help='path to sample output file.')
    parser.add_argument('--enable_random_seed', action='store_true',
                        help='is enable the random seed.')
    parser.add_argument('--skip_end_tokens', type=int, default=0,
                        help='Whether to remove or not end tokens in outputs.')
    parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                        help='Whether to compute the cumulative log probsbility of sentences.'
                             ' 0: do not return the cumulative log probs '
                             ' 1: return the cumulative log probs of generated sequences'
                             ' 2: return the cumulative log probs of sequences')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.ckpt_path, 'config.ini'))
    ## params about model
    head_num = int(config.get('llama', 'head_num'))
    size_per_head = int(config.get('llama', 'size_per_head'))
    inter_size = int(config.get('llama', 'inter_size'))
    vocab_size = int(config.get('llama', 'vocab_size'))
    layer_num = int(config.get('llama', 'num_layer'))
    rotary_embedding = int(config.get('llama', 'rotary_embedding'))
    layernorm_eps = float(config.get('llama', 'layernorm_eps'))
    start_id = int(config.get('llama', 'start_id'))
    end_id = int(config.get('llama', 'end_id'))
    use_gptj_residual = False
    weight_data_type = config.get('llama', 'weight_data_type')
    inference_data_type = args.inference_data_type
    max_seq_len = args.max_seq_len
    
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    
    # max_batch_size = args.max_batch_size
    ckpt_path = args.ckpt_path
    tokenizer_dir = args.tokenizer_dir
    
    ## params about request
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    repetition_penalty = args.repetition_penalty
    return_cum_log_probs = True 
    return_output_length = True
    output_len = args.total_seq_len
    
    print("\n=============== Arguments ===============")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")
    
    ## init tokenizer
    assert os.path.exists(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
    # eos_token_id = int(tokenizer("</MOBAN>", )['input_ids'][1])
    # tokenizer.eos_token_id = eos_token_id
    if end_id != int(tokenizer("</MOBAN>", )['input_ids'][1]):
        end_id = int(tokenizer("</MOBAN>", )['input_ids'][1])

    ## init model
    rank = 0 
    device_count = 1
    device = rank % device_count
    torch.cuda.set_device(device)
    device = torch.cuda.current_device()
    # Prepare model.
    llama = LlamaV1(head_num, size_per_head, inter_size, vocab_size, rotary_embedding, layernorm_eps,
                  start_id, end_id, layer_num, max_seq_len,
                  tensor_para_size, pipeline_para_size,
                  use_gptj_residual, None,
                  inference_data_type=inference_data_type,
                  weights_data_type=weight_data_type)
    if not llama.load(ckpt_path=ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")

    assert os.path.exists(args.sample_input_file)
    prompts = json.load(open(args.sample_input_file, 'r'))
    if args.sample_output_file:
        out_info = open(args.sample_output_file, 'w')
    batch_size = 1
    for i, prompt in enumerate(prompts):
        if args.enable_random_seed == True:
            random_seed_tensor = torch.randint(0, 10000, size=[batch_size], dtype=torch.int64)
        else:
            setup_seeds(args.seed)
            random_seed_tensor = torch.ones([batch_size], dtype=torch.int64) * args.seed
        contexts = [prompt]
        start_ids = tokenizer(contexts, return_tensors='pt').input_ids.to(torch.int32)
        max_input_len = start_ids.shape[1]
        start_lengths = torch.IntTensor([max_input_len] * batch_size)  
        request_output_len = output_len - max_input_len
        
        with torch.no_grad():
            tokens_batch = llama(
                start_ids=start_ids,
                start_lengths=start_lengths,
                output_len=request_output_len,
                beam_width=beam_width,
                top_k=top_k * torch.ones(size=[batch_size], dtype=torch.int32),
                top_p=top_p * torch.ones(size=[batch_size], dtype=torch.float32),
                beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(size=[batch_size], dtype=torch.float32),
                temperature=temperature * torch.ones(size=[batch_size], dtype=torch.float32),
                len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                repetition_penalty=repetition_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                random_seed=random_seed_tensor,
                return_output_length=return_output_length,
                return_cum_log_probs=return_cum_log_probs)
            ## if The returned results have been selected
            if return_cum_log_probs > 0:
                tokens_batch, tokens_batch_lenghts, cum_log_probs = tokens_batch
                print('[INFO] Log probs of sentences:', cum_log_probs)
            else:
                tokens_batch, tokens_batch_lenghts = tokens_batch
            true_sample_len =  tokens_batch_lenghts[0][0]   
            if rank == 0:
                tokens = tokens_batch[:,:,:true_sample_len].cpu().numpy()[0]
                context = contexts[0]
                if args.sample_output_file:
                    out_info.write(f'--------------------- 问题: {i} ------------------------\n')
                    out_info.write(context)
                    out_info.write('\n')
                for beam_id in range(beam_width):
                    token = tokens[beam_id][start_lengths[0]:]  # exclude context input from the output
                    output = tokenizer.decode(token).split("</MOBAN>")[0] if args.skip_end_tokens else tokenizer.decode(token)
                    print(f'[INFO] batch {0}, beam {beam_id}:\n[Context]\n{context}\n\n[Generated]\n{token}\n\n[Output]\n{output}\n')
                    if args.sample_output_file:
                        out_info.write(f'---------------------  回答: {i} beam_id:{beam_id}  Log probs:{cum_log_probs[0][beam_id]}------------------------\n')
                        out_info.write(output)
                        out_info.write('\n')
                        out_info.write('---------------------------------------------------------------------\n')
        if i > 10:
            break
    if args.sample_output_file:
        out_info.close()

    # Measure inference time. batch size = 1
    if args.time:
        batch_size = 1
        contexts = [prompts[0]]
        start_ids = tokenizer(contexts, return_tensors='pt').input_ids.to(torch.int32)
        max_input_len = start_ids.shape[1]
        start_lengths = torch.IntTensor([max_input_len] * batch_size)  
        request_output_len = output_len - max_input_len
        with torch.no_grad():
            iterations = 10
            # warmup
            for i in range(iterations):
                tokens_batch = llama(
                    start_ids=start_ids,
                    start_lengths=start_lengths,
                    output_len=request_output_len,
                    beam_width=beam_width,
                    top_k=top_k * torch.ones(size=[batch_size], dtype=torch.int32),
                    top_p=top_p * torch.ones(size=[batch_size], dtype=torch.float32),
                    beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(size=[batch_size], dtype=torch.float32),
                    temperature=temperature * torch.ones(size=[batch_size], dtype=torch.float32),
                    len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    repetition_penalty=repetition_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    random_seed=random_seed_tensor,
                    return_output_length=True,
                    return_cum_log_probs=0)
            batch_num = 0
            token_num = 0
            time = timeit.default_timer()
            for i in range(iterations):
                tokens_batch = llama(
                    start_ids=start_ids,
                    start_lengths=start_lengths,
                    output_len=request_output_len,
                    beam_width=beam_width,
                    top_k=top_k * torch.ones(size=[batch_size], dtype=torch.int32),
                    top_p=top_p * torch.ones(size=[batch_size], dtype=torch.float32),
                    beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(size=[batch_size], dtype=torch.float32),
                    temperature=temperature * torch.ones(size=[batch_size], dtype=torch.float32),
                    len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    repetition_penalty=repetition_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    random_seed=random_seed_tensor,
                    return_output_length=True,
                    return_cum_log_probs=0)
                tokens_batch, tokens_batch_lenghts = tokens_batch
                true_sample_len =  tokens_batch_lenghts[0][0]   
                tokens_batch = tokens_batch[:,:,:true_sample_len]
                batch_num += 1
                for j, tokens in enumerate(tokens_batch):
                    token_num += tokens.shape[-1] - start_lengths[j]
            time_elapsed = timeit.default_timer() - time
            throughput = token_num / time_elapsed
            print(f"[INFO] FT-LLAMA generates {batch_num} batches, taking {time_elapsed:0.3f} secs "
                    f"to generate {token_num} tokens, {throughput:0.3f} tokens/sec.")


if __name__ == "__main__":
    main()
    
    
    
    