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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

const int32_t NDIMS = 16;

struct SIZE_V {
  int32_t val[NDIMS];
};

struct VIS {
  bool val[NDIMS] = {false};
};

template<typename T>
void FlipCpuForward(const int32_t element, const int64_t total_dims, const SIZE_V sizes_v,
                    const VIS vis, SIZE_V strides_v, const T* in_dptr, T* out_dptr) {
  for (int i = 0; i < element; i++) {
    int32_t cur_indices = i;
    int32_t rem = 0;
    int32_t dst_offset = 0;
    for (int32_t d = 0; d < total_dims; d++) {
      int32_t temp = cur_indices;
      cur_indices = cur_indices / strides_v.val[d];
      rem = temp - cur_indices * strides_v.val[d];
      dst_offset += vis.val[d] ? (sizes_v.val[d] - 1 - cur_indices) * strides_v.val[d]
                               : cur_indices * strides_v.val[d];
      cur_indices = rem;
    }
    out_dptr[i] = in_dptr[dst_offset];
  }
}

}  // namespace

template<typename T>
class FlipCpuKernel final : public user_op::OpKernel {
 public:
  FlipCpuKernel() = default;
  ~FlipCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = y_tensor->shape_view().elem_cnt();
    if (elem_cnt == 0) { return; }
    const int32_t total_dims = y_tensor->shape_view().NumAxes();

    std::vector<int32_t> dims = ctx->Attr<std::vector<int32_t>>("dims");
    VIS vis;
    for (auto x : dims) { vis.val[x] = true; }

    SIZE_V sizes_v;
    for (int32_t i = 0; i < total_dims; i++) { sizes_v.val[i] = y_tensor->shape_view().At(i); }

    // TODO(bbuf) delete strides caluculate, after tensor strides supported
    SIZE_V strides_v;
    strides_v.val[total_dims - 1] = 1;
    for (int32_t i = total_dims - 2; i >= 0; i--) {
      strides_v.val[i] = strides_v.val[i + 1] * y_tensor->shape_view().At(i + 1);
    }

    FlipCpuForward(elem_cnt, total_dims, sizes_v, vis, strides_v, x_tensor->dptr<T>(),
                   y_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FLIP_CPU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("flip").SetCreateFn<FlipCpuKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_FLIP_CPU_KERNEL(bool)
REGISTER_FLIP_CPU_KERNEL(float)
REGISTER_FLIP_CPU_KERNEL(double)
REGISTER_FLIP_CPU_KERNEL(uint8_t)
REGISTER_FLIP_CPU_KERNEL(int8_t)
REGISTER_FLIP_CPU_KERNEL(int32_t)
REGISTER_FLIP_CPU_KERNEL(int64_t)

}  // namespace oneflow
