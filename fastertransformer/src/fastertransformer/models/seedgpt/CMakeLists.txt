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

add_library(SeedGPTDecoderLayerWeight STATIC SeedGPTDecoderLayerWeight.cc)
set_property(TARGET SeedGPTDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SeedGPTDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SeedGPTDecoderLayerWeight PUBLIC memory_utils calibrate_quantize_weight_kernels transpose_int8_kernels cuda_utils logger)

add_library(SeedGPTWeight STATIC SeedGPTWeight.cc)
set_property(TARGET SeedGPTWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SeedGPTWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SeedGPTWeight PUBLIC SeedGPTDecoderLayerWeight cuda_utils logger)

add_library(SeedGPTContextDecoder STATIC SeedGPTContextDecoder.cc)
set_property(TARGET SeedGPTContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SeedGPTContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SeedGPTContextDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer TensorParallelReluFfnLayer
                                                TensorParallelGptContextAttentionLayer layernorm_kernels
                                                add_residual_kernels bert_preprocess_kernels nccl_utils gpt_kernels tensor
                                                nvtx_utils cuda_utils logger)

add_library(SeedGPTDecoder STATIC SeedGPTDecoder.cc)
set_property(TARGET SeedGPTDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SeedGPTDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SeedGPTDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer TensorParallelReluFfnLayer
                                                TensorParallelDecoderSelfAttentionLayer layernorm_kernels
                                                add_residual_kernels nccl_utils tensor cuda_utils logger)

add_library(SeedGPT STATIC SeedGPT.cc)
set_property(TARGET SeedGPT PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SeedGPT PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SeedGPT PUBLIC -lcudart SeedGPTDecoder SeedGPTContextDecoder decoding_kernels gpt_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels gen_relative_pos_bias SeedGPTWeight
                      custom_ar_comm logprob_kernels cuda_utils logger nvtx_utils)

# add_executable(gpt_gemm gpt_gemm.cc)
# target_link_libraries(gpt_gemm PUBLIC -lcudart gpt_gemm_func memory_utils cuda_utils logger)
