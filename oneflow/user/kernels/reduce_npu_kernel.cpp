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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/user/ops/npu_command.h"
#ifdef WITH_NPU
namespace oneflow {

namespace {
template<typename dtype>
class ReduceSumNpuKernel final : public user_op::OpKernel {
 public:
  ReduceSumNpuKernel() = default;
  ~ReduceSumNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    //user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
    std::cout<<"ReduceSum "<<std::endl; 
    for(auto& i:axis) std::cout<<i<<" ";
    std::cout<<std::endl;
    if (input_tensor->shape().elem_cnt() == 0) {
      if (output_tensor->shape().elem_cnt() != 0) {
        Memset<DeviceType::kNPU>(
            ctx->stream(), output_tensor->mut_dptr<void>(), 0,
            output_tensor->shape().elem_cnt() * GetSizeOfDataType(output_tensor->data_type()));
      }
      return;
    }
    NpuCommand npu_command;
    npu_command.OpName("Reduction")
               .Input(input_tensor,"channel_last")
               .Output(output_tensor,"channel_last")
               .Attr("operation",(int64_t)1)
               .Attr("axis",(int64_t)0)
               .Attr("coeff",(float)1.0)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    PrintResult(output_tensor);
    std::cout<<"Execute Over"<<std::endl; 
               
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
}  // namespace
#define REGISTER_REDUCE_NPU_KERNEL(op_name, dtype)                            \
  REGISTER_USER_KERNEL(op_name)                                                                    \
      .SetCreateFn<ReduceSumNpuKernel<dtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() ==  DeviceType::kNPU)                                        \
                       && (user_op::HobDataType("output_tensor", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                          \
        const Shape& in_shape = ctx->InputShape("input_tensor", 0);                                \
        return in_shape.elem_cnt() * sizeof(dtype);                                                \
      });

REGISTER_REDUCE_NPU_KERNEL("reduce_sum",float);
REGISTER_REDUCE_NPU_KERNEL("reduce_sum",float16);
} // namespace oneflow
#endif