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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/slice_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/ops/npu_command.h"

#define HOSTWRAPPER_FROM_VECTOR(v)    \
    std::vector<int64_t> v ## _desc = {static_cast<int>(v ## _attr.size())}; \
    HostTensorWrapper v ## _wrap(ACL_INT32, ACL_FORMAT_ND, v ## _desc.size(), v ## _desc.data(), \
                            v ## _attr.size()*sizeof(int), v ## _attr.data());  
namespace oneflow {

template<typename T>
class SliceGradNpuKernel final : public user_op::OpKernel {
 public:
  SliceGradNpuKernel() = default;
  ~SliceGradNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t dx_byte_size = dx_tensor->shape_view().elem_cnt() * sizeof(T);
    return ;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SLICE_KERNEL(device, dtype)                                               \
  REGISTER_USER_KERNEL("slice_grad")                                                       \
      .SetCreateFn<SliceGradNpuKernel<dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));   \

#define REGISTER_SLICE_NPU_KERNEL_WITH_DEVICE(device) \
  REGISTER_SLICE_KERNEL(device, bool)             \
  REGISTER_SLICE_KERNEL(device, float)            \
  REGISTER_SLICE_KERNEL(device, float16)            \
  REGISTER_SLICE_KERNEL(device, double)           \
  REGISTER_SLICE_KERNEL(device, int32_t)          \
  REGISTER_SLICE_KERNEL(device, int64_t)          \
  REGISTER_SLICE_KERNEL(device, int8_t)           \
  REGISTER_SLICE_KERNEL(device, uint8_t)

REGISTER_SLICE_NPU_KERNEL_WITH_DEVICE(DeviceType::kNPU)

}  // namespace oneflow
