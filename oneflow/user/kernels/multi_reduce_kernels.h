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
#ifndef ONEFLOW_USER_KERNELS_MULTI_REDUCE_KERNELS_H_
#define ONEFLOW_USER_KERNELS_MULTI_REDUCE_KERNELS_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/multi_reduce_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MultiReduceSumPowAbsKernel final : public user_op::OpKernel,
                                         public user_op::CudaGraphSupport {
 public:
  MultiReduceSumPowAbsKernel() = default;
  ~MultiReduceSumPowAbsKernel() override = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache*) const override {
    std::vector<MultiReduceParam<T>> params;
    params.resize(ctx->input_size("x"));
    for (size_t i = 0; i < params.size(); ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", i);
      params[i].size = x->shape_view().elem_cnt();
      params[i].data = x->dptr<T>();
    }
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    T* y_dptr = y->mut_dptr<T>();
    user_op::Tensor* temp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* tmp_dptr = temp ? temp->mut_dptr<T>() : nullptr;
    float p = ctx->Attr<float>("p");
    if (p == 0) {
      PowByZero<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), y_dptr, tmp_dptr);
    } else if (p == 1) {
      Abs<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), y_dptr, tmp_dptr);
    } else if (p == 2) {
      Square<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), y_dptr, tmp_dptr);
    } else {
      AbsPow<T> func{p};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), y_dptr, tmp_dptr);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

enum class Ximum {
  kMax = 0,
  kMin = 1,
};

template<DeviceType device_type, typename T, Ximum X>
class MultiReduceXimumAbsKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MultiReduceXimumAbsKernel() = default;
  ~MultiReduceXimumAbsKernel() override = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache*) const override {
    std::vector<MultiReduceParam<T>> params;
    params.resize(ctx->input_size("x"));
    for (size_t i = 0; i < params.size(); ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", i);
      params[i].size = x->shape_view().elem_cnt();
      params[i].data = x->dptr<T>();
    }
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* temp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* tmp_dptr = temp ? temp->mut_dptr<T>() : nullptr;
    Abs<T> abs{};
    if (X == Ximum::kMax) {
      MultiReduce<device_type, T, decltype(abs), BinaryMax<T>> reduce_max{};
      reduce_max(ctx->stream(), abs, params, GetZeroVal<T>(), y->mut_dptr<T>(), tmp_dptr);
    } else if (X == Ximum::kMin) {
      MultiReduce<device_type, T, decltype(abs), BinaryMin<T>> reduce_min{};
      reduce_min(ctx->stream(), abs, params, std::numeric_limits<T>::max(), y->mut_dptr<T>(),
                 tmp_dptr);
    } else {
      UNIMPLEMENTED();
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_MULTI_REDUCE_KERNELS_H_
