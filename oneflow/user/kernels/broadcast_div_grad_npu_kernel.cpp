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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace {

template<DeviceType device>
class BroadcastDivGradNpuKernel final : public user_op::OpKernel {
 public:
  BroadcastDivGradNpuKernel() = default;
  ~BroadcastDivGradNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("z", 0);
    user_op::Tensor* dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    AclTensorWrapper wrap(nullptr, ACL_FLOAT, 0, nullptr, ACL_FORMAT_ND, sizeof(float));
    NpuCommand div_command;
    div_command.OpName("RealDiv")
                .Input(z)
                .Input(y)
                .Output(wrap)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
    div_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    NpuCommand mul_command;
    mul_command.OpName("Mul")
                .Input(dz)
                .Input(wrap)
                .Output(dy)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
    div_command.Run();
    //PrintResult(dy);
    //std::cout<<"MathBinaryBroadcastAddKernel Execute Over"<<std::endl; 

  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BROADCAST_DIV_GRAD_NPU_KERNEL(device, dtype)                             \
  REGISTER_USER_KERNEL("broadcast_div_grad")                                               \
      .SetCreateFn<BroadcastDivGradNpuKernel<device>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));
    //   .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                         
    //     const user_op::TensorDesc& z = ctx->InputTensorDesc("z", 0);                       
    //     const DataType& data_type = z.data_type();                                         
    //     const int64_t elem_cnt = z.shape().elem_cnt();                                     
    //     return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type));                
    //   });
REGISTER_BROADCAST_DIV_GRAD_NPU_KERNEL(kNPU, float);
REGISTER_BROADCAST_DIV_GRAD_NPU_KERNEL(kNPU, float16);
}  // namespace oneflow
