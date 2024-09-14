/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/layers/attention_layers_fp8/AttentionFP8Weight.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

template<typename T>
struct GroupedQuery_attention_params: public Multihead_attention_params_base<T> {
    // allows to exist attention eary
    bool* finished     = nullptr;
    int   num_kv_heads = 0;
    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;

    float rope_theta;
    float rope_scaling;
};

template<class T>
using Masked_groupedquery_attention_params = GroupedQuery_attention_params<T>;

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_groupedquery_attention(const Masked_groupedquery_attention_params<float>& params,
                                   const cudaStream_t&                                stream);
void masked_groupedquery_attention(const Masked_groupedquery_attention_params<uint16_t>& params,
                                   const cudaStream_t&                                   stream);
#ifdef ENABLE_BF16
void masked_groupedquery_attention(const Masked_groupedquery_attention_params<__nv_bfloat16>& params,
                                   const cudaStream_t&                                        stream);
#endif
#ifdef ENABLE_FP8
void masked_groupedquery_attention(const Masked_groupedquery_attention_params<__nv_fp8_e4m3>& params,
                                   const cudaStream_t&                                        stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace fastertransformer {

template<typename T>
struct BaseAttentionLayout {
    int    stride_batch;
    int    stride_seq;
    int    stride_head;
    bool   use_seqlens       = false;
    size_t batch_seqs_offset = 0;
    T**    batch_seqs        = nullptr;
};

template<typename T>
struct BaseAttentionParams {
    T*                     attn_out;
    T*                     query;
    T*                     key;
    T*                     val;
    T*                     mask;
    float*                 out_accum       = nullptr;
    int*                   cu_seqlens_q    = nullptr;
    int*                   cu_seqlens_k    = nullptr;
    int*                   actual_seqlen_q = nullptr;
    int*                   actual_seqlen_k = nullptr;
    size_t                 group_size      = 1;
    BaseAttentionLayout<T> layout_q;
    BaseAttentionLayout<T> layout_k;
    BaseAttentionLayout<T> layout_v;
    BaseAttentionLayout<T> layout_o;
};

template<typename T, int version>
class FlashAttentionOpImpl {
public:
    using AttentionLayout = BaseAttentionLayout<T>;
    using Params          = BaseAttentionParams<T>;

public:
    FlashAttentionOpImpl(int batch_size, int head_num, int key_len, int seq_len, int size_per_head);
    ~FlashAttentionOpImpl();

    int get_workspace_size() const;

    void operator()(Params& params, cudaStream_t st) const;

private:
    class impl;
    std::unique_ptr<impl> pimpl;
};

template<typename T>
class FlashAttentionOp {
public:
    using AttentionLayout = BaseAttentionLayout<T>;
    using Params          = BaseAttentionParams<T>;

public:
    FlashAttentionOp(int batch_size, int head_num, int key_len, int seq_len, int size_per_head);

    int get_workspace_size() const;

    void operator()(Params& params, cudaStream_t st) const;

private:
    int batch_size_;
    int head_num_;
    int key_len_;
    int seq_len_;
    int size_per_head_;
    int op_version_;
};

}