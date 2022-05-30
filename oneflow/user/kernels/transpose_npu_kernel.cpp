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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace user_op {

namespace {
bool IsIdentity(const std::vector<int32_t>& perm) {
  for (auto i = 0; i < perm.size(); i++) {
    if (perm[i] != i) { return false; }
  }
  return true;
}
}  // namespace

class TransposeNpuKernel final : public OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransposeNpuKernel);
  TransposeNpuKernel() = default;
  ~TransposeNpuKernel() override = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {


    Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("input", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("output", 0);
    std::vector<int32_t> perm = ctx->Attr<std::vector<int32_t>>("perm");
    const ShapeView& in_shape = tensor_in->shape();
    DataType dtype = tensor_out->data_type();
    size_t num_dims = tensor_in->shape().NumAxes();
    const int64_t* src_dims = in_shape.ptr();

    int64_t elem_cnt = tensor_out->shape().elem_cnt();

    if (elem_cnt != 0) {
      if (IsIdentity(perm)) {
        // if permute vector is 0,1,...,n, do data copy directly
        AutoMemcpy(ctx->stream(), tensor_out->mut_dptr(), tensor_in->dptr(),
                   elem_cnt * GetSizeOfDataType(dtype), tensor_out->mem_case(),
                   tensor_in->mem_case());
        //std::cout<<"Transpose aclrtMemcpy over"<<std::endl;
      } else {
          std::vector<int64_t> shape_desc = {static_cast<int64_t>(perm.size())};
          AclTensorWrapper wrap(nullptr, ACL_INT32, shape_desc.size(), shape_desc.data(), ACL_FORMAT_ND,
                                  mulVector(perm)*sizeof(int32_t), perm.data());
          NpuCommand npu_command;
          npu_command.OpName("Transpose")
                    .Input(tensor_in)
                    .Input(wrap)
                    .Output(tensor_out)
                    .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                    .Check();
          npu_command.Run();
          OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));  
          //PrintResult(tensor_out);
          //std::cout<<"TransposeNpuKernel Execute Over"<<std::endl;  
      }

    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


REGISTER_USER_KERNEL("transpose")
    .SetCreateFn<TransposeNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);

}  // namespace user_op
}  // namespace oneflow
