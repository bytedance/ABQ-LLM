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

#include "src/fastertransformer/kernels/llama/decoder_masked_groupedquery_attention.h"
#include "src/fastertransformer/kernels/llama/decoder_masked_groupedquery_attention/decoder_masked_groupedquery_attention_template.hpp"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

template<typename T, typename KERNEL_PARAMS_TYPE>
void groupedquery_attention_(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream)
{
    switch (params.hidden_size_per_head) {
        case 32:
            launch_mgqa_kernel<T, 32, 32, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 48:
            launch_mgqa_kernel<T, 48, 64, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 64:
            launch_mgqa_kernel<T, 64, 64, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 80:
            launch_mgqa_kernel<T, 80, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 96:
            launch_mgqa_kernel<T, 96, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 128:
            launch_mgqa_kernel<T, 128, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 144:
            launch_mgqa_kernel<T, 144, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 160:
            launch_mgqa_kernel<T, 160, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 192:
            launch_mgqa_kernel<T, 192, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 224:
            launch_mgqa_kernel<T, 224, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 256:
            launch_mgqa_kernel<T, 256, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        default:
            assert(false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_groupedquery_attention(const Masked_groupedquery_attention_params<float>& params, const cudaStream_t& stream)
{
    groupedquery_attention_<float, Masked_groupedquery_attention_params<float>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_groupedquery_attention(const Masked_groupedquery_attention_params<uint16_t>& params, const cudaStream_t& stream)
{
    groupedquery_attention_<uint16_t, Masked_groupedquery_attention_params<uint16_t>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
void masked_groupedquery_attention(const Masked_groupedquery_attention_params<__nv_bfloat16>& params,
                                const cudaStream_t&                                     stream)
{
    groupedquery_attention_<__nv_bfloat16, Masked_groupedquery_attention_params<__nv_bfloat16>>(params, stream);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
void masked_groupedquery_attention(const Masked_groupedquery_attention_params<__nv_fp8_e4m3>& params,
                                const cudaStream_t&                                     stream)
{
    groupedquery_attention_<__nv_fp8_e4m3, Masked_groupedquery_attention_params<__nv_fp8_e4m3>>(params, stream);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace fastertransformer {

#define VERSION_SWITCH(VERSION, CONST_NAME, ...)                                                                       \
    [&] {                                                                                                              \
        if (VERSION == 2) {                                                                                            \
            constexpr static int CONST_NAME = 2;                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else {                                                                                                         \
            constexpr static int CONST_NAME = 1;                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

template<typename T>
FlashAttentionOp<T>::FlashAttentionOp(int batch_size, int head_num, int key_len, int seq_len, int size_per_head):
    batch_size_(batch_size), head_num_(head_num), key_len_(key_len), seq_len_(seq_len), size_per_head_(size_per_head)
{
#ifdef _MSC_VER
    op_version_ = 1;
#else
    op_version_ = std::is_same<half, typename std::decay<T>::type>::value ? 2 : 1;
    if (op_version_ == 2 && getSMVersion() < 80) {
        op_version_ = 1;
    }

#endif
}

template<typename T>
int FlashAttentionOp<T>::get_workspace_size() const
{
#ifdef _MSC_VER
    FlashAttentionOpImpl<T, 1> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
    return attention_op.get_workspace_size();
#else
    return VERSION_SWITCH(op_version_, OP_VERSION, [&]() {
        FlashAttentionOpImpl<T, OP_VERSION> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
        return attention_op.get_workspace_size();
    });
#endif
}

template<typename T>
void FlashAttentionOp<T>::operator()(Params& params, cudaStream_t st) const
{

#ifdef _MSC_VER
    FlashAttentionOpImpl<T, 1> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
    return attention_op(params, st);
#else
    return VERSION_SWITCH(op_version_, OP_VERSION, [&]() {
        FlashAttentionOpImpl<T, OP_VERSION> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
        return attention_op(params, st);
    });
#endif
}

template class FlashAttentionOp<float>;
template class FlashAttentionOp<half>;
#ifdef ENABLE_BF16
template class FlashAttentionOp<__nv_bfloat16>;
#endif
}