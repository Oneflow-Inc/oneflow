/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_USER_KERNELS_MULTI_TENSOR_MODEL_UPDATE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_MULTI_TENSOR_MODEL_UPDATE_KERNEL_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

// Kernel arg size has 4K limit, but currently we set process 32 tensors in each kernel.
constexpr int kMaxTuples = 32;

template<int N>
struct TensorTupleParams {
  void* ptr[N][kMaxTuples];
  int64_t sizes[kMaxTuples];
  int32_t block_offset[kMaxTuples];
};

template<DeviceType device_type, typename T, typename G>
struct MultiTensorSGDUpdateKernelUtil {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, TensorTupleParams<2> tensor_tuple_params);
};

template<DeviceType device_type, typename T, typename G>
struct MultiTensorMomentumUpdateKernelUtil {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const float momentum, const float dampening,
                     const bool nesterov, const bool maximize,
                     TensorTupleParams<3> tensor_tuple_params);
};

template<DeviceType device_type, typename T, typename G>
struct MultiTensorAdamUpdateKernelUtil {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float beta1, float beta2, float epsilon,
                     float weight_decay, bool amsgrad, bool do_bias_correction,
                     float learning_rate_val, float bias_correction1_val,
                     float bias_correction2_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const float* bias_correction1,
                     const float* bias_correction2, TensorTupleParams<4> tensor_tuple_params);
};

template<DeviceType device_type, typename T, typename G>
struct MultiTensorSGDUpdateWithCastKernelUtil {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, TensorTupleParams<3> tensor_tuple_params);
};

template<DeviceType device_type, typename T, typename G>
struct MultiTensorMomentumUpdateWithCastKernelUtil {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const float momentum, const float dampening,
                     const bool nesterov, const bool maximize,
                     TensorTupleParams<4> tensor_tuple_params);
};

template<DeviceType device_type, typename T, typename G>
struct MultiTensorAdamUpdateWithCastKernelUtil {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float beta1, float beta2, float epsilon,
                     float weight_decay, bool amsgrad, bool do_bias_correction,
                     float learning_rate_val, float bias_correction1_val,
                     float bias_correction2_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const float* bias_correction1,
                     const float* bias_correction2, TensorTupleParams<5> tensor_tuple_params);
};

template<DeviceType device_type, typename T>
struct MultiTensorYoloV5WeightUpdateKernelUtil {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, float d,
                     TensorTupleParams<2> tensor_tuple_params);
};

}  // namespace oneflow

#endif
