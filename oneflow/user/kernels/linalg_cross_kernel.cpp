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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

template<typename T>
class CpuLinalgCrossKernel final : public user_op::OpKernel {
 public:
  CpuLinalgCrossKernel() = default;
  ~CpuLinalgCrossKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* other_tensor = ctx->Tensor4ArgNameAndIndex("other", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    const auto shape = input_tensor->shape_view();
    const auto num_axes = shape.NumAxes();

    int64_t dim = ctx->Attr<int64_t>("dim");

    const auto strides = [&shape]() -> std::vector<int64_t> {
      std::vector<int64_t> result(shape.NumAxes(), 1);
      for (size_t i(0); i < result.size() - 1; ++i) { result[i] = shape.Count(i + 1); }
      return result;
    }();

    const int64_t total = shape.elem_cnt() / 3;
    int64_t stride = strides[dim];

    const T* input_ptr = input_tensor->dptr<T>();
    const T* other_ptr = other_tensor->dptr<T>();
    T* out_dtr = out_tensor->mut_dptr<T>();

    std::vector<int64_t> positions_in_dims(num_axes);

    int64_t start = 0;

    int64_t s = 0;
    while (s < total) {
      out_dtr[start + 0 * stride] = input_ptr[start + 1 * stride] * other_ptr[start + 2 * stride]
                                    - input_ptr[start + 2 * stride] * other_ptr[start + 1 * stride];
      out_dtr[start + 1 * stride] = input_ptr[start + 2 * stride] * other_ptr[start + 0 * stride]
                                    - input_ptr[start + 0 * stride] * other_ptr[start + 2 * stride];
      out_dtr[start + 2 * stride] = input_ptr[start + 0 * stride] * other_ptr[start + 1 * stride]
                                    - input_ptr[start + 1 * stride] * other_ptr[start + 0 * stride];

      ++s;

      FOR_RANGE(int64_t, i, 0, num_axes) {
        if (i == dim) continue;

        ++positions_in_dims[i];
        start += strides[i];

        if (positions_in_dims[i] == shape.At(i) && i != num_axes - 1) {
          start -= positions_in_dims[i] * strides[i];
          positions_in_dims[i] = 0;
        } else {
          break;
        }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_LINALG_CROSS_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("linalg_cross")                                \
      .SetCreateFn<CpuLinalgCrossKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value));

REGISTER_CPU_LINALG_CROSS_KERNEL(float)
REGISTER_CPU_LINALG_CROSS_KERNEL(double)

}  // namespace oneflow