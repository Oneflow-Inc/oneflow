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
#include "oneflow/user/kernels/avg_pooling_kernel_util.h"
#include "oneflow/user/ops/npu_command.h"

namespace oneflow {

class AvgPool2dNpuKernel final : public user_op::OpKernel {
 public:
  AvgPool2dNpuKernel() = default;
  ~AvgPool2dNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    std::string data_format = ctx->Attr<std::string>("data_format");
    std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
    std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("stride");
    std::vector<int64_t> kernel_64 = {1, 1, kernel_size[0], kernel_size[1]};
    std::vector<int64_t> stride_64 = {1, 1, stride[0], stride[1]};
    //std::vector<int64_t> pad_64 = {padding[0], padding[0], padding[1], padding[1]};
    std::vector<int64_t> pad_64 = {0, 0, 0, 0};
    VECTOR_PRINT(kernel_64);
    VECTOR_PRINT(stride_64);
    VECTOR_PRINT(pad_64);
    bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    bool count_include_pad = ctx->Attr<bool>("count_include_pad");// dck_caution_here
    int32_t divisor_override = ctx->Attr<int32_t>("divisor_override");
    
     NpuCommand npu_command;
     npu_command.OpName("AvgPoolV2")
                .Input(x, "channel_first")
                .Output(y, "channel_first")
                .Attr("ksize", kernel_64)
                .Attr("strides", stride_64)
                .Attr("padding_mode", std::string("CALCULATED"))
                .Attr("pads", pad_64)
                .Attr("data_format", data_format)
                .Attr("global_pooling", false)
                .Attr("ceil_mode", ceil_mode)
                .Attr("exclusive", false)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
      npu_command.Run();
      OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
      PrintResult(y);
      std::cout<<"Execute Over"<<std::endl; 
  };
};
#define REGISTER_AVG_POOLING_KERNELS(dtype)                                           \
    REGISTER_USER_KERNEL("avgpool_2d")                                                \
    .SetCreateFn<AvgPool2dNpuKernel>()                                                \
    .SetIsMatchedHob((user_op::HobDeviceType() == kNPU)                               \
                    && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));  \

REGISTER_AVG_POOLING_KERNELS(float);
REGISTER_AVG_POOLING_KERNELS(float16);
} // namespace oneflow