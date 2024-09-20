# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(LlamaV2DecoderLayerWeight STATIC LlamaV2DecoderLayerWeight.cc)
set_property(TARGET LlamaV2DecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaV2DecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaV2DecoderLayerWeight PUBLIC memory_utils cuda_utils logger)

add_library(LlamaV2Decoder STATIC LlamaV2Decoder.cc)
set_property(TARGET LlamaV2Decoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaV2Decoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaV2Decoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelLlamaV2DecoderSelfAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      LlamaV2DecoderLayerWeight
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(LlamaV2ContextDecoder STATIC LlamaV2ContextDecoder.cc)
set_property(TARGET LlamaV2ContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaV2ContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaV2ContextDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelLlamaV2ContextAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      gpt_kernels
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(LlamaV2Weight STATIC LlamaV2Weight.cc)
set_property(TARGET LlamaV2Weight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaV2Weight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaV2Weight PUBLIC LlamaV2DecoderLayerWeight cuda_utils logger)

add_library(LlamaV2 STATIC LlamaV2.cc)
set_property(TARGET LlamaV2 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaV2 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaV2 PUBLIC -lcudart
                      LlamaV2Decoder
                      LlamaV2ContextDecoder
                      decoding_kernels
                      gpt_kernels
                      DynamicDecodeLayer
                      BaseBeamSearchLayer
                      bert_preprocess_kernels
                      tensor
                      LlamaV2Weight
                      cuda_utils
                      logger)
