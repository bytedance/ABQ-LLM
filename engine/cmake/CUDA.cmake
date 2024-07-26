# Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if(CUDA_COMPILER MATCHES "[Cc]lang")
  set(BYTELLM_NATIVE_CUDA_INIT ON)
elseif(CMAKE_VERSION VERSION_LESS 3.12.4)
  set(BYTELLM_NATIVE_CUDA_INIT OFF)
else()
  set(BYTELLM_NATIVE_CUDA_INIT ON)
endif()

set(BYTELLM_NATIVE_CUDA ${BYTELLM_NATIVE_CUDA_INIT} CACHE BOOL "Utilize the CMake native CUDA flow")

if(NOT DEFINED ENV{CUDACXX} AND NOT DEFINED ENV{CUDA_BIN_PATH} AND DEFINED ENV{CUDA_PATH})
  # For backward compatibility, allow use of CUDA_PATH.
  set(ENV{CUDACXX} $ENV{CUDA_PATH}/bin/nvcc)
endif()

if(BYTELLM_NATIVE_CUDA)
  message(STATUS "Using BYTELLM_NATIVE_CUDA")
  enable_language(CUDA)
  if(NOT CUDA_VERSION)
    set(CUDA_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
  endif()
  if(NOT CUDA_TOOLKIT_ROOT_DIR)
    get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
  endif()

else()
  find_package(CUDA REQUIRED)
  # We workaround missing variables with the native flow by also finding the CUDA toolkit the old way.
  if(NOT CMAKE_CUDA_COMPILER_VERSION)
    set(CMAKE_CUDA_COMPILER_VERSION ${CUDA_VERSION})
  endif()

endif()

if (CUDA_VERSION VERSION_LESS 10.2)
  message(FATAL_ERROR "CUDA 10.2+ Required, Found ${CUDA_VERSION}.")
endif()
if(NOT BYTELLM_NATIVE_CUDA OR CUDA_COMPILER MATCHES "[Cc]lang")
  set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
  message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
endif()
