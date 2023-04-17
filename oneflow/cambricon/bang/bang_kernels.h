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
#ifndef ONEFLOW_CAMBRICON_BANG_BANG_KERNELS_H_
#define ONEFLOW_CAMBRICON_BANG_BANG_KERNELS_H_

#include "oneflow/cambricon/bang/bang_handle.h"

namespace oneflow {

// input is a 3D tensor with shape [batch, N, length]
// indices is a 1D tensor with shape [index_size]
// output is a 3D tensor with shape [batch, index_size, length]
template<typename T, typename K>
void bang_gather_kernel(BangHandle& handle, const T* input, int64_t batch, int64_t N,
                        int64_t length, const K* index, int64_t index_size, T* output,
                        int64_t offset);

template<typename K>
void bang_gather_half_kernel(BangHandle& handle, const void* input, int64_t batch, int64_t N,
                             int64_t length, const K* index, int64_t index_size, void* output,
                             int64_t offset);

// input is a 3D tensor with shape [batch, segment_size, length]
// indices is a 1D tensor with shape [segment_size]
// output is a 3D tensor with shape [batch, N, length]
template<typename T, typename K>
void bang_unsorted_segment_sum_kernel(BangHandle& handle, const T* input, int64_t batch, int64_t N,
                                      int64_t length, const K* segment, int64_t segment_size,
                                      T* output, int64_t offset);

template<typename K>
void bang_unsorted_segment_sum_half_kernel(BangHandle& handle, const void* input, int64_t batch,
                                           int64_t N, int64_t length, const K* segment,
                                           int64_t segment_size, void* output, int64_t offset);

template<typename T>
void bang_momentum_update_kernel(BangHandle& handle, int64_t n, T scale, float l1, float l2,
                                 float beta, float dampening, bool nesterov, bool maximize,
                                 float weight_decay, float learning_rate, float lr_scale,
                                 const float* learning_rate_ptr, const T* scale_by_ptr,
                                 const int64_t* skip_if, const T* model_diff, T* model,
                                 T* momentum);

template<typename T>
void bang_momentum_update_half_kernel(BangHandle& handle, int64_t n, T scale, float l1, float l2,
                                      float beta, float dampening, bool nesterov, bool maximize,
                                      float weight_decay, float learning_rate, float lr_scale,
                                      const float* learning_rate_ptr, const T* scale_by_ptr,
                                      const int64_t* skip_if, const void* model_diff, T* model,
                                      T* momentum);

template<typename T>
void bang_adam_update_kernel(BangHandle& handle, int64_t n, T scale, float l1, float l2,
                             float beta1, float beta2, float epsilon, float weight_decay,
                             bool amsgrad, bool do_bias_correction, float learning_rate,
                             float lr_scale, float bias_correction1_val, float bias_correction2_val,
                             const float* learning_rate_ptr, const T* scale_by_ptr,
                             const int64_t* skip_if, const float* bias_correction1_ptr,
                             const float* bias_correction2_ptr, const T* model_diff, T* model,
                             void* model_copy, T* m, T* v, T* max_v);

template<typename T>
void bang_adam_update_half_kernel(BangHandle& handle, int64_t n, T scale, float l1, float l2,
                                  float beta1, float beta2, float epsilon, float weight_decay,
                                  bool amsgrad, bool do_bias_correction, float learning_rate,
                                  float lr_scale, float bias_correction1_val,
                                  float bias_correction2_val, const float* learning_rate_ptr,
                                  const T* scale_by_ptr, const int64_t* skip_if,
                                  const float* bias_correction1_ptr,
                                  const float* bias_correction2_ptr, const void* model_diff,
                                  T* model, void* model_copy, T* m, T* v, T* max_v);

template<typename T>
void bang_regularize_gradient_kernel(BangHandle& handle, int64_t n, const T* model,
                                     const T* model_diff, T* out, float l1, float l2);

void bang_regularize_gradient_half_kernel(BangHandle& handle, int64_t n, const void* model,
                                          const void* model_diff, void* out, float l1, float l2);

template<typename T>
void bang_multi_reduce_sum_pow_abs_kernel(BangHandle& handle, int64_t n, const T** inputs,
                                          const int64_t* sizes, T* output, float p, void* workspace,
                                          int64_t workspace_size);

template<typename T>
void bang_multi_count_not_finite_kernel(BangHandle& handle, int64_t n, const T** inputs,
                                        const int64_t* sizes, int64_t* output, void* workspace,
                                        int64_t workspace_size);

void bang_multi_count_not_finite_half_kernel(BangHandle& handle, int64_t n, const void** inputs,
                                             const int64_t* sizes, int64_t* output, void* workspace,
                                             int64_t workspace_size);

template<typename T>
void bang_scalar_pow_gradient_kernel(BangHandle& handle, int64_t n, const T* x, const T* dy,
                                     const float value, T* dx);

void bang_scalar_pow_gradient_half_kernel(BangHandle& handle, int64_t n, const void* x,
                                          const void* dy, const float value, void* dx);

template<typename T>
void bang_tanh_gradient_kernel(BangHandle& handle, int64_t n, const T* x, const T* dy, T* dx);

void bang_tanh_gradient_half_kernel(BangHandle& handle, int64_t n, const void* x, const void* dy,
                                    void* dx);

template<typename T>
void bang_fast_gelu_kernel(BangHandle& handle, int64_t n, const T* in, T* out);

void bang_fast_gelu_half_kernel(BangHandle& handle, int64_t n, const void* in, void* out);

template<typename T>
void bang_fast_gelu_grad_kernel(BangHandle& handle, int64_t n, const T* out_grad, const T* in,
                                T* in_grad);

void bang_fast_gelu_grad_half_kernel(BangHandle& handle, int64_t n, const void* out_grad,
                                     const void* in, void* in_grad);

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_BANG_BANG_KERNELS_H_
