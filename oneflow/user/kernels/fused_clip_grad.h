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
#ifndef ONEFLOW_USER_KERNELS_FUSED_CLIP_GRAD_H_
#define ONEFLOW_USER_KERNELS_FUSED_CLIP_GRAD_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/multi_reduce_kernel_util.h"
#include "oneflow/user/kernels/fused_clip_grad_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class FusedClipGradKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  FusedClipGradKernel() = default;
  ~FusedClipGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* out_ptr = out->mut_dptr<T>();
    T* temp = (ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0))->mut_dptr<T>();
    const int32_t input_size = ctx->input_size("model_diff");
    const float max_norm = ctx->Attr<float>("max_norm");
    const float norm_type = ctx->Attr<float>("norm_type");

    std::vector<MultiReduceParam<T>> params;
    params.resize(input_size);
    for (size_t i = 0; i < input_size; ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("model_diff", i);
      params[i].size = x->shape_view().elem_cnt();
      params[i].data = x->dptr<T>();
    }
    if (norm_type == 0) {
      PowByZero<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_add{};
      reduce_add(ctx->stream(), func, params, GetZeroVal<T>(), out_ptr, temp);
    } else if (norm_type == INFINITY) {
      Abs<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryMax<T>> reduce_max{};
      reduce_max(ctx->stream(), func, params, GetZeroVal<T>(), out_ptr, temp);
    } else if (norm_type == -INFINITY) {
      Abs<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryMin<T>> reduce_min{};
      reduce_min(ctx->stream(), func, params, std::numeric_limits<T>::max(), out_ptr, temp);
    } else if (norm_type == 1) {
      Abs<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), out_ptr, temp);
    } else if (norm_type == 2) {
      Square<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), out_ptr, temp);
    } else {
      AbsPow<T> func{norm_type};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), out_ptr, temp);
    }

    std::vector<MultiClipGradParam<T>> mut_params;
    mut_params.resize(input_size);
    for (size_t i = 0; i < input_size; ++i) {
      user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("model_diff", i);
      mut_params[i].size = x->shape_view().elem_cnt();
      mut_params[i].data = x->mut_dptr<T>();
    }
    MultiClipGrad<device_type, T> multi_clip_grad{};
    if (norm_type == 0) {
      multi_clip_grad(ctx->stream(), mut_params, out_ptr, norm_type, max_norm,
                      ClipGradType::ZeroType);
    } else if (std::abs(norm_type) == INFINITY || norm_type == 1) {
      multi_clip_grad(ctx->stream(), mut_params, out_ptr, norm_type, max_norm,
                      ClipGradType::OtherType);
    } else {
      multi_clip_grad(ctx->stream(), mut_params, out_ptr, norm_type, max_norm,
                      ClipGradType::PowerType);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_FUSED_CLIP_GRAD_H_
