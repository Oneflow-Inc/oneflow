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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/roll_kernel_utils.h"

#include <algorithm>

namespace oneflow {

template<typename T>
class CpuRollKernel final : public user_op::OpKernel {
 public:
  CpuRollKernel() = default;
  ~CpuRollKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::vector<int32_t>& shifts = ctx->Attr<std::vector<int32_t>>("shifts");
    const std::vector<int32_t>& dims = ctx->Attr<std::vector<int32_t>>("dims");

    SHAPE new_shape{};
    SHIFTS new_shifts{};
    int32_t num_axes = 0;
    computeParams(in->shape_view(), shifts, dims, new_shifts.val, new_shape.val, &num_axes);

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    const int32_t size = out->shape_view().elem_cnt();

    STRIDE stride{};
    initStride(stride, new_shape, num_axes);

    transformShifts(new_shifts.val, new_shape.val, num_axes);

    for (int32_t i = 0; i < size; ++i) {
      int shifted_i = switchGetShiftedIndex(i, new_shifts.val, new_shape.val, stride.val, num_axes);
      out_ptr[i] = in_ptr[shifted_i];
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ROLL_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("roll").SetCreateFn<CpuRollKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_ROLL_KERNEL(float);
REGISTER_ROLL_KERNEL(double);
REGISTER_ROLL_KERNEL(bool);
REGISTER_ROLL_KERNEL(uint8_t);
REGISTER_ROLL_KERNEL(int8_t);
REGISTER_ROLL_KERNEL(int32_t);
REGISTER_ROLL_KERNEL(int64_t);

}  // namespace oneflow
