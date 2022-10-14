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
#include "oneflow/user/kernels/gather_kernel_util.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/user/ops/npu_command.h"

namespace oneflow {

namespace user_op {

template<typename T, typename K>
class GatherNpuKernel final : public user_op::OpKernel {
 public:
  GatherNpuKernel() = default;
  ~GatherNpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    int axis = ctx->Attr<int64_t>("axis");
    const int64_t num_indices = indices->shape_view().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out->shape_view().elem_cnt() == 0) { return; }

    std::vector<int64_t> axis_desc = {1};
    HostTensorWrapper wrap(ACL_INT32, ACL_FORMAT_ND, axis_desc.size(), axis_desc.data(),
                            sizeof(int), &axis);

    NpuCommand npu_command;
    npu_command.OpName("GatherV2")
               .Input(in)
               .Input(indices)
               .Input(wrap)
               .Output(out)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GATHER_NPU_KERNEL(in_type, indices_type)                                \
  REGISTER_USER_KERNEL("gather")                                                             \
      .SetCreateFn<                                                                          \
          GatherNpuKernel<in_type, indices_type>>() \
      .SetIsMatchedHob(                                                                      \
          (user_op::HobDeviceType() == DeviceType::kNPU)                                               \
          && (user_op::HobDataType("in", 0) == GetDataType<in_type>::value)                   \
          && (user_op::HobDataType("indices", 0) == GetDataType<indices_type>::value));


REGISTER_GATHER_NPU_KERNEL(float, int);
REGISTER_GATHER_NPU_KERNEL(float16, int);

}  // namespace user_op

}  // namespace oneflow
