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
#include "oneflow/core/ep/include/primitive/add.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace user_op {

namespace {

class AddNNpuKernel : public OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddNNpuKernel);
  AddNNpuKernel() = default;
  ~AddNNpuKernel() override = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType data_type = out->data_type();
    const size_t count = out->shape().elem_cnt();
    size_t in_num = ctx->inputs().size();
    CHECK_EQ(in_num, 2)<<"Current only support AddV2, so input num should be 2. ";
    std::vector<const void*> srcs(in_num);
    NpuCommand npu_command;
    for (size_t i = 0; i < in_num; ++i) {
      user_op::Tensor* in_i = ctx->Tensor4ArgNameAndIndex("in", i);
      CHECK_EQ(in_i->shape().elem_cnt(), count);
      CHECK_EQ(in_i->data_type(), data_type);
      npu_command.Input(in_i, "channels_nd");
    }
    npu_command.OpName("AddV2")
               //.Attr("N", (int64_t)in_num)
               .Output(out, "channels_nd")
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    //PrintResult(out);
    //std::cout<<"AddN Execute Over"<<std::endl; 
  }
};

REGISTER_USER_KERNEL("add_n")
    .SetCreateFn<AddNNpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU))
    .SetInplaceProposalFn([](const InferContext&,
                             const AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace user_op

}  // namespace oneflow
