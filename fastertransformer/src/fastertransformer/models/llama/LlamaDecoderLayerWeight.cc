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

#include "src/fastertransformer/models/llama/LlamaDecoderLayerWeight.h"
#include "src/fastertransformer/kernels/transpose_int8_kernels.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(const int  hidden_units,
                                                   const int  inter_size,
                                                   const int  tensor_para_size,
                                                   const int  tensor_para_rank,
                                                   const bool use_gptj_residual,
                                                   const int  int8_mode):
   hidden_units_(hidden_units),
   inter_size_(inter_size),
   tensor_para_size_(tensor_para_size),
   tensor_para_rank_(tensor_para_rank),
   int8_mode_(int8_mode),
   use_gptj_residual_(use_gptj_residual)
{
   mallocWeights();
   setWeightPtr();

//    FT_CHECK_WITH_INFO(int8_mode_ != 2, "Llama doesn't support int8_model == 2");
   FT_CHECK_WITH_INFO(!(std::is_same<T, float>::value && (int8_mode_ == 1 || int8_mode_ == 4)),
                      "Weight only quant does not work with FP32 compute.");
}

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(const int int8_mode): int8_mode_(int8_mode)
{
}

template<typename T>
LlamaDecoderLayerWeight<T>::~LlamaDecoderLayerWeight()
{
   if (is_maintain_buffer == true) {
       for (int i = 0; i < weights_ptr.size(); i++) {
           if (weights_ptr[i] != nullptr) {
               deviceFree(weights_ptr[i]);
           }
       }

       pre_layernorm_weights.beta                            = nullptr;
       pre_layernorm_weights.gamma                           = nullptr;
       self_attention_weights.query_weight.kernel            = nullptr;
       self_attention_weights.query_weight.bias              = nullptr;
       self_attention_weights.attention_output_weight.kernel = nullptr;
       self_attention_weights.attention_output_weight.bias   = nullptr;
       post_attention_layernorm_weights.beta                 = nullptr;
       post_attention_layernorm_weights.gamma                = nullptr;

       ffn_weights.intermediate_weight.kernel = nullptr;
       ffn_weights.intermediate_weight.bias   = nullptr;
       ffn_weights.intermediate_weight2.kernel = nullptr;
       ffn_weights.intermediate_weight2.bias   = nullptr;
       ffn_weights.output_weight.kernel       = nullptr;
       ffn_weights.output_weight.bias         = nullptr;

       if (int8_mode_ != 0) {
           for (size_t i = 0; i < int8_weights_ptr.size(); i++) {
               if (int8_weights_ptr[i] != nullptr) {
                   deviceFree(int8_weights_ptr[i]);
               }
           }

           if (int8_mode_ == 1 || int8_mode_ == 4) {
               for (size_t i = 0; i < weight_only_scale_ptr.size(); i++) {
                   if (weight_only_scale_ptr[i] != nullptr) {
                       deviceFree(weight_only_scale_ptr[i]);
                   }
               }
           } 
           else if(int8_mode_ == 2 || int8_mode_ == 5) {
                for (size_t i = 0; i < scale_ptr.size(); i++){  
                    if(scale_ptr[i] != nullptr){
                        deviceFree(scale_ptr[i]);
                    }
                }
                for (size_t i = 0; i < scale_inter_ptr.size(); i++){  
                    if(scale_inter_ptr[i] != nullptr){
                        deviceFree(scale_inter_ptr[i]);
                    }
                }
                for (size_t i = 0; i < scale_out_ptr.size(); i++){  
                    if(scale_out_ptr[i] != nullptr){
                        deviceFree(scale_out_ptr[i]);
                    }
                }

           }

           self_attention_weights.query_weight.int8_kernel                             = nullptr;
           self_attention_weights.query_weight.weight_only_quant_scale                 = nullptr;
           self_attention_weights.query_weight.scale                                   = nullptr;
           self_attention_weights.query_weight.scale_inter                             = nullptr;
           self_attention_weights.query_weight.scale_out                               = nullptr;
           self_attention_weights.attention_output_weight.int8_kernel                  = nullptr;
           self_attention_weights.attention_output_weight.weight_only_quant_scale      = nullptr;
           self_attention_weights.attention_output_weight.scale                                   = nullptr;
           self_attention_weights.attention_output_weight.scale_inter                             = nullptr;
           self_attention_weights.attention_output_weight.scale_out                               = nullptr;
           // NOTE: intermediate_weight => gate_proj;  intermediate_weight2 => up_proj; output_weight => down_proj.
           ffn_weights.intermediate_weight.int8_kernel                                 = nullptr;
           ffn_weights.intermediate_weight.weight_only_quant_scale                     = nullptr;
           ffn_weights.intermediate_weight.scale                                   = nullptr;
           ffn_weights.intermediate_weight.scale_inter                             = nullptr;
           ffn_weights.intermediate_weight.scale_out                               = nullptr;

           ffn_weights.intermediate_weight2.int8_kernel                                = nullptr;
           ffn_weights.intermediate_weight2.weight_only_quant_scale                    = nullptr;
           ffn_weights.intermediate_weight2.scale                                   = nullptr;
           ffn_weights.intermediate_weight2.scale_inter                             = nullptr;
           ffn_weights.intermediate_weight2.scale_out                               = nullptr;
           ffn_weights.output_weight.int8_kernel                                       = nullptr;
           ffn_weights.output_weight.weight_only_quant_scale                           = nullptr;
           ffn_weights.output_weight.scale                                   = nullptr;
           ffn_weights.output_weight.scale_inter                             = nullptr;
           ffn_weights.output_weight.scale_out                               = nullptr;

       }

       is_maintain_buffer                     = false;
   }
}

template<typename T>
void LlamaDecoderLayerWeight<T>::copyFrom(const LlamaDecoderLayerWeight& other)
{
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    if (!use_gptj_residual_) {
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    }
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);
    cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], hidden_units_);
    cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);

    if (int8_mode_ == 0) {
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    }
    else if(int8_mode_ == 5){
        // w2a8
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], (w_bits_ * hidden_units_) * (3 * hidden_units_ / sz_) / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], (w_bits_ * hidden_units_) / tensor_para_size_ * (hidden_units_ / sz_));
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], (w_bits_ * hidden_units_) * (inter_size_ / sz_) / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], (w_bits_ * hidden_units_) * (inter_size_ / sz_) / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], (w_bits_ * inter_size_) / tensor_para_size_ * (hidden_units_ / sz_));
        
        cudaD2Dcpy(scale_ptr[0], other.scale_out_ptr[0], 1);
        cudaD2Dcpy(scale_inter_ptr[0], other.scale_inter_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_out_ptr[0], other.scale_out_ptr[0], 3);

        for (int i = 1; i < 5; i++) {
            cudaD2Dcpy(scale_ptr[i], other.scale_ptr[i], 1);
            cudaD2Dcpy(scale_inter_ptr[i], other.scale_inter_ptr[i], 1);
            cudaD2Dcpy(scale_out_ptr[i], other.scale_out_ptr[i], 1);
        }
    }
    else {
        if (int8_mode_ == 4){
            cudaD2Dcpy(int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_ / 2);
            cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_ / 2);
            cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_ / 2);
            cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_ / 2);
            cudaD2Dcpy(int8_weights_ptr[4], other.int8_weights_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_ / 2);
        } else {
            cudaD2Dcpy(int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[4], other.int8_weights_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);
        }
        if (int8_mode_ == 1 || int8_mode_ == 4) {
            cudaD2Dcpy(weight_only_scale_ptr[0], other.weight_only_scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);  // query_key_value weight per channel scale
            cudaD2Dcpy(weight_only_scale_ptr[1], other.weight_only_scale_ptr[1], hidden_units_);         // attention output weight per channel scale
            cudaD2Dcpy(weight_only_scale_ptr[2], other.weight_only_scale_ptr[2], inter_size_ / tensor_para_size_); // intermediate_weight weight per channel scale (gate_proj)
            cudaD2Dcpy(weight_only_scale_ptr[3], other.weight_only_scale_ptr[3], inter_size_ / tensor_para_size_);  // intermediate_weight2 weight per channel scale (up_proj)
            cudaD2Dcpy(weight_only_scale_ptr[4], other.weight_only_scale_ptr[4], hidden_units_);                    // output_weight weight per channel scale (down_proj)
        }
        else if(int8_mode_ == 2){
            cudaD2Dcpy(scale_ptr[0], other.scale_out_ptr[0], 1);
            cudaD2Dcpy(scale_inter_ptr[0], other.scale_inter_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            cudaD2Dcpy(scale_out_ptr[0], other.scale_out_ptr[0], 3);
            for (size_t i = 1; i < 5; i++)
            {
                cudaD2Dcpy(scale_ptr[i], other.scale_ptr[i], 1);
                cudaD2Dcpy(scale_inter_ptr[i], other.scale_inter_ptr[i], 1);
                cudaD2Dcpy(scale_out_ptr[i], other.scale_out_ptr[i], 1);
            }
        }
    }
}

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(const LlamaDecoderLayerWeight& other):
   hidden_units_(other.hidden_units_),
   inter_size_(other.inter_size_),
   tensor_para_size_(other.tensor_para_size_),
   tensor_para_rank_(other.tensor_para_rank_),
   int8_mode_(other.int8_mode_),
   use_gptj_residual_(other.use_gptj_residual_)
{
   mallocWeights();
   copyFrom(other);
   setWeightPtr();
}

template<typename T>
LlamaDecoderLayerWeight<T>& LlamaDecoderLayerWeight<T>::operator=(const LlamaDecoderLayerWeight& other)
{
   hidden_units_      = other.hidden_units_;
   inter_size_        = other.inter_size_;
   tensor_para_size_  = other.tensor_para_size_;
   tensor_para_rank_  = other.tensor_para_rank_;
   int8_mode_          = other.int8_mode_;
   use_gptj_residual_ = other.use_gptj_residual_;

   mallocWeights();

   copyFrom(other);
   setWeightPtr();
   return *this;
}

template<typename T>
void LlamaDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
   FT_CHECK(is_maintain_buffer == true);
   const std::string rank_spec = std::to_string(tensor_para_rank_);

   // fill all bias to zeros
   // LLAMA`s RMSNorm doesn`t have bias
   deviceFill(weights_ptr[0], (size_t)hidden_units_, (T)0.0);
   loadWeightFromBin<T>(
       weights_ptr[1], {(size_t)hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);

   deviceFill(weights_ptr[3], (size_t)(3 * hidden_units_ / tensor_para_size_), (T)0.0);

   if (!use_gptj_residual_) {
       deviceFill(weights_ptr[5], (size_t)hidden_units_, (T)0.0);
   }

   deviceFill(weights_ptr[7], (size_t)(inter_size_ / tensor_para_size_), (T)0.0);

   deviceFill(weights_ptr[9], (size_t)(inter_size_ / tensor_para_size_), (T)0.0);

   deviceFill(weights_ptr[11], (size_t)(hidden_units_), (T)0.0);

   // LLAMA`s RMSNorm doesn`t have bias
   deviceFill(weights_ptr[12], (size_t)(hidden_units_), (T)0.0);
   loadWeightFromBin<T>(
       weights_ptr[13], {(size_t)hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);

   // Load qkv weight, attention output weight, intermediate_weight, intermediate_weight2, output weight
   if (int8_mode_ == 0) {
       loadWeightFromBin<T>(weights_ptr[2],
                            {(size_t)hidden_units_, (size_t)(3 * hidden_units_ / tensor_para_size_)},
                            dir_path + ".attention.query_key_value.weight." + rank_spec + ".bin",
                            model_file_type);

       loadWeightFromBin<T>(weights_ptr[4],
                            {(size_t)(hidden_units_ / tensor_para_size_), (size_t)hidden_units_},
                            dir_path + ".attention.dense.weight." + rank_spec + ".bin",
                            model_file_type);

       loadWeightFromBin<T>(weights_ptr[6],
                            {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                            dir_path + ".mlp.gate_proj.weight." + rank_spec + ".bin",
                            model_file_type);

       loadWeightFromBin<T>(weights_ptr[8],
                            {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                            dir_path + ".mlp.up_proj.weight." + rank_spec + ".bin",
                            model_file_type);
                            
       loadWeightFromBin<T>(weights_ptr[10],
                            {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units_},
                            dir_path + ".mlp.down_proj.weight." + rank_spec + ".bin",
                            model_file_type);
   }
   else if (int8_mode_ == 1 || int8_mode_ == 4) {
       loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[0],
                                                    weight_only_scale_ptr[0],
                                                    {(size_t)hidden_units_, (size_t)(3 * hidden_units_ / tensor_para_size_)},
                                                    dir_path + ".attention.query_key_value.weight." + rank_spec + ".bin",
                                                    model_file_type,
                                                    int8_mode_);

       loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[1],
                                                    weight_only_scale_ptr[1],
                                                    {(size_t)(hidden_units_ / tensor_para_size_), (size_t)hidden_units_},
                                                    dir_path + ".attention.dense.weight." + rank_spec + ".bin",
                                                    model_file_type,
                                                    int8_mode_);

       loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[2],
                                                    weight_only_scale_ptr[2],
                                                    {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                                                    dir_path + ".mlp.gate_proj.weight." + rank_spec + ".bin",
                                                    model_file_type,
                                                    int8_mode_);

       loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[3],
                                                    weight_only_scale_ptr[3],
                                                    {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                                                    dir_path + ".mlp.up_proj.weight." + rank_spec + ".bin",
                                                    model_file_type,
                                                    int8_mode_);
       loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[4],
                                                    weight_only_scale_ptr[4],
                                                    {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units_},
                                                    dir_path + ".mlp.down_proj.weight." + rank_spec + ".bin",
                                                    model_file_type,
                                                    int8_mode_);

   }   
   else if(int8_mode_ == 2){
        const std::vector<std::string> weight_list{
            "attention.query_key_value", "attention.dense", "mlp.gate_proj",  "mlp.up_proj", "mlp.down_proj"};
        const std::vector<std::vector<size_t>> shape_list{{size_t(hidden_units_), size_t(3 * hidden_units_ / tensor_para_size_)},
                                                          {size_t(hidden_units_ / tensor_para_size_), size_t(hidden_units_)},
                                                          {size_t(hidden_units_), size_t(inter_size_ / tensor_para_size_)},
                                                          {size_t(hidden_units_), size_t(inter_size_ / tensor_para_size_)},
                                                          {size_t(inter_size_ / tensor_para_size_), size_t(hidden_units_)}};
    
        for (int i = 0; i < weight_list.size(); i++) {
            loadWeightFromBin<int8_t>(int8_weights_ptr[i],
                                      shape_list[i],
                                      dir_path + "." + weight_list[i] + ".weight.int8." + rank_spec + ".bin",
                                      FtCudaDataType::INT8);

            const std::pair<std::vector<std::vector<float*>*>, std::vector<std::string>> arg_pair{
                {&scale_ptr, &scale_inter_ptr, &scale_out_ptr}, {"scale", "scale_inter", "scale_out"}};
            for (int j = 0; j < arg_pair.first.size(); j++) {
                size_t num_elems = 1;
                // attention.qkv scale_inter has 3 weights for Q, K and V
                // attention.qkv scale_out has 3 weights for Q, K and V, duplicated along hidden_units dim
                if (i == 0 && j == 1) {
                    num_elems = 3 * hidden_units_ / tensor_para_size_;
                }
                else if (i == 0 && j == 2) {
                    num_elems = 3;
                }

                loadWeightFromBin<float>((*arg_pair.first[j])[i],
                                         {num_elems},
                                         dir_path + "." + weight_list[i] + "." + arg_pair.second[j] + ".bin",
                                         FtCudaDataType::FP32);
            }
        }    
        transposeWeight();
   }
   else if(int8_mode_ == 5){
        loadWeightFromBin<T>(weights_ptr[2],
                         {(size_t)(w_bits_ * hidden_units_), (size_t)(3 * hidden_units_ / sz_ / tensor_para_size_)},
                         dir_path + ".attention.query_key_value.weight." + rank_spec + ".bin",
                         model_file_type);

        loadWeightFromBin<T>(weights_ptr[4],
                         {(size_t)(w_bits_ * hidden_units_ / tensor_para_size_), (size_t)(hidden_units_ / sz_)},
                         dir_path + ".attention.dense.weight." + rank_spec + ".bin",
                         model_file_type);

        loadWeightFromBin<T>(weights_ptr[6],
                         {(size_t)(w_bits_ * hidden_units_), (size_t)(inter_size_ / sz_ / tensor_para_size_)},
                         dir_path + ".mlp.gate_proj.weight." + rank_spec + ".bin",
                         model_file_type);

        loadWeightFromBin<T>(weights_ptr[8],
                         {(size_t)(inter_size_ / sz_ / tensor_para_size_), (size_t)(w_bits_ * hidden_units_)},
                         dir_path + ".mlp.up_proj.weight." + rank_spec + ".bin",
                         model_file_type);
        loadWeightFromBin<T>(weights_ptr[10],
                         {(size_t)(w_bits_ * inter_size_ / tensor_para_size_), (size_t)(hidden_units_ / sz_)},
                         dir_path + ".mlp.down_proj.weight." + rank_spec + ".bin",
                         model_file_type);
        // TODO: Load scales
    }
}

template<typename T>
void LlamaDecoderLayerWeight<T>::setWeightPtr()
{
   pre_layernorm_weights.beta                            = weights_ptr[0];
   pre_layernorm_weights.gamma                           = weights_ptr[1];
   self_attention_weights.query_weight.kernel            = weights_ptr[2];
   self_attention_weights.query_weight.bias              = weights_ptr[3];
   self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
   self_attention_weights.attention_output_weight.bias   = use_gptj_residual_ ? nullptr : weights_ptr[5];

   ffn_weights.intermediate_weight.kernel  = weights_ptr[6];
   ffn_weights.intermediate_weight.bias    = weights_ptr[7];
   ffn_weights.intermediate_weight2.kernel = weights_ptr[8];
   ffn_weights.intermediate_weight2.bias   = weights_ptr[9];
   ffn_weights.output_weight.kernel        = weights_ptr[10];
   ffn_weights.output_weight.bias          = weights_ptr[11];

   post_attention_layernorm_weights.beta  = weights_ptr[12];
   post_attention_layernorm_weights.gamma = weights_ptr[13];

   if (int8_mode_ != 0) {
       self_attention_weights.query_weight.int8_kernel                 = int8_weights_ptr[0];
       self_attention_weights.attention_output_weight.int8_kernel      = int8_weights_ptr[1];
       ffn_weights.intermediate_weight.int8_kernel                     = int8_weights_ptr[2];
       ffn_weights.intermediate_weight2.int8_kernel                    = int8_weights_ptr[3];
       ffn_weights.output_weight.int8_kernel                           = int8_weights_ptr[4];

       if (int8_mode_ == 1 || int8_mode_ == 4) {
           self_attention_weights.query_weight.weight_only_quant_scale                 = weight_only_scale_ptr[0];
           self_attention_weights.attention_output_weight.weight_only_quant_scale      = weight_only_scale_ptr[1];
           ffn_weights.intermediate_weight.weight_only_quant_scale                     = weight_only_scale_ptr[2];
           ffn_weights.intermediate_weight2.weight_only_quant_scale                    = weight_only_scale_ptr[3];
           ffn_weights.output_weight.weight_only_quant_scale                           = weight_only_scale_ptr[4];
       } 
       else if(int8_mode_ == 2 || int8_mode_ == 5){
            self_attention_weights.query_weight.scale                  = scale_ptr[0];
            self_attention_weights.query_weight.scale_inter            = scale_inter_ptr[0];
            self_attention_weights.query_weight.scale_out              = scale_out_ptr[0];
            self_attention_weights.attention_output_weight.scale       = scale_ptr[1];
            self_attention_weights.attention_output_weight.scale_inter = scale_inter_ptr[1];
            self_attention_weights.attention_output_weight.scale_out   = scale_out_ptr[1];
            ffn_weights.intermediate_weight.scale                      = scale_ptr[2];
            ffn_weights.intermediate_weight.scale_inter                = scale_inter_ptr[2];
            ffn_weights.intermediate_weight.scale_out                  = scale_out_ptr[2];
            ffn_weights.intermediate_weight2.scale                      = scale_ptr[3];
            ffn_weights.intermediate_weight2.scale_inter                = scale_inter_ptr[3];
            ffn_weights.intermediate_weight2.scale_out                  = scale_out_ptr[3];
            ffn_weights.output_weight.scale                            = scale_ptr[4];
            ffn_weights.output_weight.scale_inter                      = scale_inter_ptr[4];
            ffn_weights.output_weight.scale_out                        = scale_out_ptr[4];
       }
   }
   is_maintain_buffer                     = true;
}

template<typename T>
void LlamaDecoderLayerWeight<T>::mallocWeights()
{
   deviceMalloc(&weights_ptr[0], hidden_units_); // pre layernorm beta
   deviceMalloc(&weights_ptr[1], hidden_units_); // pre layernorm gamma
   // deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_); // qkv kernel
   deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_); // qkv bias
   // deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_); // attention output weight
   if (!use_gptj_residual_) {
       deviceMalloc(&weights_ptr[5], hidden_units_); // attention output bias
   }

   // deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_); // intermediate_weight kernel
   deviceMalloc(&weights_ptr[7], inter_size_ / tensor_para_size_);                 // intermediate_weight bias
   // deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_); // intermediate_weight2 kernel
   deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);                 // intermediate_weight2 bias
   // deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_); // output_weight kernel
   deviceMalloc(&weights_ptr[11], hidden_units_);                                   // output_weight bias
   deviceMalloc(&weights_ptr[12], hidden_units_); // post attn layernorm beta
   deviceMalloc(&weights_ptr[13], hidden_units_); // post attn layernorm gamma

   if (int8_mode_ == 0) {
       deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);  // qkv weight
       deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);  // attention output weight
       deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);   // intermediate_weight kernel
       deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);  // intermediate_weight2 kernel
       deviceMalloc(&weights_ptr[10],  inter_size_ / tensor_para_size_ * hidden_units_);  // output_weight kernel
   }
   else if(int8_mode_ == 5){
        // Alloc FFN and Attention weights
        deviceMalloc(&weights_ptr[2], (w_bits_ * hidden_units_) * (3 * hidden_units_ / sz_) / tensor_para_size_);  // qkv weight
        deviceMalloc(&weights_ptr[4], (w_bits_ * hidden_units_) / tensor_para_size_ * (hidden_units_ / sz_));  // attention output weight
        
        deviceMalloc(&weights_ptr[6], (w_bits_ * hidden_units_) * (inter_size_ / sz_) / tensor_para_size_);   // intermediate_weight kernel
        deviceMalloc(&weights_ptr[8], (w_bits_ * hidden_units_) * (inter_size_ / sz_) / tensor_para_size_);  // intermediate_weight2 kernel
        deviceMalloc(&weights_ptr[10],  (w_bits_ * inter_size_) / tensor_para_size_ * (hidden_units_ / sz_));  // output_weight kernel

        // Alloc scale for weight activation quant
        deviceMalloc(&scale_ptr[0], 1);
        deviceMalloc(&scale_inter_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&scale_out_ptr[0], 3);
        for (int i = 1; i < 5; i++) {
            deviceMalloc(&scale_ptr[i], 1);
            deviceMalloc(&scale_inter_ptr[i], 1);
            deviceMalloc(&scale_out_ptr[i], 1);
        }
    }
   else {
       // Alloc FFN and Attention int8 weights
       if (int8_mode_ == 4){
            // The weight is quantized using int4, but is folded into the data type of int8 
            // (the two adjacent numbers are located in the lower 4 bits and the higher 4 bits respectively)
            deviceMalloc(&int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_ / 2);
            deviceMalloc(&int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_ / 2);
            deviceMalloc(&int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_ / 2);
            deviceMalloc(&int8_weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_ / 2);
            deviceMalloc(&int8_weights_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_ / 2);
       } 
       else{
            deviceMalloc(&int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
            deviceMalloc(&int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
            deviceMalloc(&int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
            deviceMalloc(&int8_weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
            deviceMalloc(&int8_weights_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);
       }

       if (int8_mode_ == 1 || int8_mode_ == 4) {
           // Alloc scales for weight only quant for attention and FFN weights
           deviceMalloc(&weight_only_scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
           deviceMalloc(&weight_only_scale_ptr[1], hidden_units_);
           deviceMalloc(&weight_only_scale_ptr[2], inter_size_ / tensor_para_size_);
           deviceMalloc(&weight_only_scale_ptr[3], inter_size_ / tensor_para_size_);
           deviceMalloc(&weight_only_scale_ptr[4], hidden_units_);
       } 
       else if(int8_mode_ == 2){
            deviceMalloc(&scale_ptr[0], 1);
            deviceMalloc(&scale_inter_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            deviceMalloc(&scale_out_ptr[0], 3);
            for (int i = 1; i < 5; i++) {
                deviceMalloc(&scale_ptr[i], 1);
                deviceMalloc(&scale_inter_ptr[i], 1);
                deviceMalloc(&scale_out_ptr[i], 1);
            }
       }
   }
}

#ifdef SPARSITY_ENABLED

template<typename T>
void LlamaDecoderLayerWeight<T>::compress_weight(cublasMMWrapper& cublas_wrapper, int hidden_dim){
   compress_weight(cublas_wrapper, hidden_dim, false, 0, 0);
}

template<typename T>
void LlamaDecoderLayerWeight<T>::compress_weight(cublasMMWrapper& cublas_wrapper, int hidden_dim, const bool release_old,
    const int layer_index, const int tp_rank){
    
    hidden_units_ = hidden_dim;
    inter_size_ = 4 * hidden_units_;
    const size_t num_sparse_weights = 5;
    const auto type_size = sizeof(T);

    size_t weight_shape[num_sparse_weights][2] = {
        {hidden_units_, 3*hidden_units_ / tensor_para_size_},
        {hidden_units_ / tensor_para_size_, hidden_units_},
        {hidden_units_ , inter_size_ / tensor_para_size_},
        {hidden_units_ , inter_size_ / tensor_para_size_},
        {inter_size_ / tensor_para_size_, hidden_units_}
    };
    const std::vector<std::string> weights_name{
        "attention.query_key_value", "attention.dense", "mlp.gate_proj",  "mlp.up_proj", "mlp.down_proj"};

    const T* dense_weights[num_sparse_weights] = {
        self_attention_weights.query_weight.kernel,
        self_attention_weights.attention_output_weight.kernel,
        ffn_weights.intermediate_weight.kernel,
        ffn_weights.intermediate_weight2.kernel,
        ffn_weights.output_weight.kernel
    };
    size_t real_num_sparse_weights = num_sparse_weights;
    for (size_t i = 0; i < real_num_sparse_weights; i++)
    {
        int k = weight_shape[i][0];
        int m = weight_shape[i][1];
        size_t compressd_size = cublas_wrapper.getSparseMatrixSize(m, k);
        size_t old_size= type_size * m * k;
        std::string weight_name = weights_name[i];
        
        FT_LOG_INFO("Sparse Compression weight {%s} TP {%d} Layer {%d}: {%ld} --> {%ld} [%d,%d](bytes)", weight_name.c_str(), tp_rank, layer_index,
            old_size, compressd_size, k, m);

        deviceMalloc(&sp_weights_ptr[i], static_cast<int>(compressd_size), false);
        cublas_wrapper.compressMatrix(reinterpret_cast<const void *>(dense_weights[i]), sp_weights_ptr[i], m, k);
        if (release_old && dense_weights[i] != nullptr){
            T* release_point = const_cast<T*>(dense_weights[i]);
            deviceFree(release_point);
        }
    }
}

#endif // SPARSITY_ENABLED

template<typename T>
void LlamaDecoderLayerWeight<T>::transposeWeight()
{
    const auto                             tp = tensor_para_size_;
    const std::vector<std::vector<size_t>> shape_list{{size_t(hidden_units_), size_t(3 * hidden_units_ / tp)},
                                                      {size_t(hidden_units_ / tp), size_t(hidden_units_)},
                                                      {size_t(hidden_units_), size_t(inter_size_ / tp)},
                                                      {size_t(hidden_units_), size_t(inter_size_ / tp)},
                                                      {size_t(inter_size_ / tp), size_t(hidden_units_)}};

    const auto max_size =
        sizeof(int8_t) * std::max(3 * hidden_units_ * hidden_units_ / tp, hidden_units_ * inter_size_ / tp);

    int8_t* transpose_temp;
    cudaMalloc(&transpose_temp, max_size);

    for (int i = 0; i < shape_list.size(); i++) {
        // k*m row_major --> m*k row_major
        invokeTransposeInt8Tensor({MEMORY_GPU, TYPE_INT8, {shape_list[i][1], shape_list[i][0]}, transpose_temp},
                                  {MEMORY_GPU, TYPE_INT8, shape_list[i], int8_weights_ptr[i]},
                                  stream_);
        cudaD2Dcpy(int8_weights_ptr[i], transpose_temp, shape_list[i][0] * shape_list[i][1]);
    }

    cudaFree(transpose_temp);
}

template struct LlamaDecoderLayerWeight<float>;
template struct LlamaDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template class LlamaDecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
