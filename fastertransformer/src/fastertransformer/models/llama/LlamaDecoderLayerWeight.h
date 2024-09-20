/*
* Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <string>

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"

namespace fastertransformer {

struct llamaVariantParams {
    // GPT default params
    float          layernorm_eps   = 1e-6f;
    LayerNormType  layernorm_type  = LayerNormType::pre_layernorm;
    ActivationType activation_type = ActivationType::Silu;
    // Whether to have a learnt positional encoding.
    bool has_positional_encoding = false;
    // A layernom just after the word embedding and before the decoder.
    bool has_pre_decoder_layernorm = false;
    // A layernom after the decoder.
    bool has_post_decoder_layernorm = true;
    // detoxification adapters. refer to
    bool   has_adapters       = false;
    size_t adapter_inter_size = 0;
    // Whether to use the attention linear positional bias
    bool use_attention_linear_bias = false;
    // Whether to use gptj residual add mode
    bool use_gptj_residual = false;
};


template<typename T>
struct LlamaDecoderLayerWeight {
public:
   LlamaDecoderLayerWeight() = default;
   LlamaDecoderLayerWeight(const int int8_mode);
   LlamaDecoderLayerWeight(const int  hidden_units,
                           const int  inter_size,
                           const int  tensor_para_size  = 1,
                           const int  tensor_para_rank  = 0,
                           const bool use_gptj_residual = true,
                           const int int8_mode = 0);
   ~LlamaDecoderLayerWeight();
   LlamaDecoderLayerWeight(const LlamaDecoderLayerWeight& other);
   LlamaDecoderLayerWeight& operator=(const LlamaDecoderLayerWeight& other);

   void loadModel(std::string dir_path, FtCudaDataType model_file_type);

#ifdef SPARSITY_ENABLED
   void compress_weight(cublasMMWrapper& cublas_wrapper, int hidden_dim);
   void compress_weight(cublasMMWrapper& cublas_wrapper, int hidden_dim, const bool release_old, 
                         const int layer_index, const int tp_rank);
#endif
   void transposeWeight();

   LayerNormWeight<T> pre_layernorm_weights;
   AttentionWeight<T> self_attention_weights;
   LayerNormWeight<T> post_attention_layernorm_weights;
   FfnWeight<T>       ffn_weights;

private:
   void setWeightPtr();
   void mallocWeights();
   void copyFrom(const LlamaDecoderLayerWeight& other);

protected:
   int       hidden_units_;
   int       inter_size_;
   int       tensor_para_size_ = 1;
   int       tensor_para_rank_ = 0;
   bool      use_gptj_residual_;
   const int attention_dense_bias_weight_id = 5;
   bool      is_maintain_buffer             = false;
   int       int8_mode_ = 0;
   int       x_bits_ = 8;
   int       w_bits_ = 2;
   const int       sz_     = sizeof(T) * 8;
   std::vector<T*> weights_ptr = std::vector<T*>(14, nullptr);
   std::vector<int8_t*> int8_weights_ptr = std::vector<int8_t*>(5, nullptr);
   std::vector<T*>      weight_only_scale_ptr = std::vector<T*>(5, nullptr);

   // // int8_mode == 2 
   // Qx = x * Scalex; Qy = y * Scaley; Qw = w * Scalew; 
   // x*w -->y == Qx*Qw*(Scaley/(Scalex*Scalew)) = Qy
   // scale_inter = Scaley/(Scalex*Scalew)
   // scale_out = 1/Scaley, it means: Qy*scale_out = y
   // scale = Scalex, it means: x * Scalex = Qx
   std::vector<float*> scale_ptr = std::vector<float*>(5, nullptr);
   std::vector<float*> scale_inter_ptr = std::vector<float*>(5, nullptr);
   std::vector<float*> scale_out_ptr = std::vector<float*>(5, nullptr);

   cudaStream_t stream_ = 0;

#ifdef SPARSITY_ENABLED
   std::vector<T*> sp_weights_ptr = std::vector<T*>(5, nullptr);
   bool is_maintrain_sp_buffer = false;
#endif // SPARSITY_ENABLED

};

}  // namespace fastertransformer
