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
#ifndef _ONEFLOW_USER_KERNELS_MASKED_FORK_H_
#define _ONEFLOW_USER_KERNELS_MASKED_FORK_H_
#include <cstdint>
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
struct ForkLoopFunctor {
  OF_DEVICE_FUNC void operator()(const int64_t elem_cnt, const T* in, const int8_t* mask,
                                 T* out_true, T* out_false) {
    XPU_1D_KERNEL_LOOP(i, elem_cnt) {
      T out_true_val = 0;
      T out_false_val = 0;
      if (mask[i]) {
        out_true_val = in[i];
      } else {
        out_false_val = in[i];
      }
      if (out_true) { out_true[i] = out_true_val; }
      if (out_false) { out_false[i] = out_false_val; }
    }
  }
};

template<DeviceType device_type, typename T>
struct MaskedForkFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, const T* in, const int8_t* mask, T* out_true,
                  const T* out_false);
};

template<DeviceType device_type, typename T>
class MaskedForkKernel final : public user_op::OpKernel {
 public:
  MaskedForkKernel() = default;
  ~MaskedForkKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* tensor_mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* tensor_out_true = ctx->Tensor4ArgNameAndIndex("out_true", 0);
    user_op::Tensor* tensor_out_false = ctx->Tensor4ArgNameAndIndex("out_false", 0);

    const T* dptr_in = tensor_in->dptr<T>();
    const int8_t* dptr_mask = tensor_mask->dptr<int8_t>();

    T* dptr_out_true = tensor_out_true ? tensor_out_true->mut_dptr<T>() : nullptr;
    T* dptr_out_false = tensor_out_false ? tensor_out_false->mut_dptr<T>() : nullptr;

    MaskedForkFunctor<device_type, T>()(ctx->device_ctx(), tensor_in->shape().elem_cnt(), dptr_in,
                                        dptr_mask, dptr_out_true, dptr_out_false);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MASKED_FORK_KERNEL(device, dtype)                                    \
  REGISTER_USER_KERNEL("masked_fork")                                                 \
      .SetCreateFn<MaskedForkKernel<device, dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                            \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("mask", 0) == GetDataType<int8_t>::value))
}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_MASKED_FORK_H_
