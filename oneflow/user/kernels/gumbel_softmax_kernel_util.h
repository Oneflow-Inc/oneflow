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
#ifndef ONEFLOW_USER_KERNELS_GUMBEL_SOFTMAX_UTIL_H_
#define ONEFLOW_USER_KERNELS_GUMBEL_SOFTMAX_UTIL_H_
#ifdef WITH_CUDA
#include "oneflow/core/ep/cuda/cuda_stream.h"
#endif  // WITH_CUDA
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/ep/include/primitive/softmax.h"
#include "oneflow/core/ep/include/primitive/softmax_backward.h"

namespace oneflow {

template<typename Context>
std::unique_ptr<ep::primitive::Softmax> NewSoftmaxPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::SoftmaxFactory>(ctx->device_type(), data_type);
}

template<typename Context>
std::unique_ptr<ep::primitive::SoftmaxBackward> NewSoftmaxBackwardPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::SoftmaxBackwardFactory>(ctx->device_type(),
                                                                            data_type);
}

template<DeviceType device_type, typename T>
struct GumbelSoftmaxAddNoiseImpl final {
  static void Forward(ep::Stream* stream, double tau, int64_t elem_cnt, const T* in_ptr,
                      T* gumbel_noise_ptr, T* out_ptr);
};

#define INITIATE_GUMBEL_SOFTMAX_KERNEL_UTIL_IMPL(device_type_v, dtype_pair) \
  template struct GumbelSoftmaxAddNoiseImpl<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

#define GUMBEL_SOFTMAX_KERNEL_DATA_TYPE_SEQ ARITHMETIC_DATA_TYPE_SEQ

}  //  namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_GUMBEL_SOFTMAX_UTIL_H_
