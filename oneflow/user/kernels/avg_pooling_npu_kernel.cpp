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
    std::vector<int64_t> pad_64 = {padding[0], padding[0], padding[1], padding[1]};

    bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    bool count_include_pad = ctx->Attr<bool>("count_include_pad");// dck_caution_here
    int32_t divisor_override = ctx->Attr<int32_t>("divisor_override");
    
     NpuCommand npu_command;
     npu_command.OpName("AvgPoolV2")
                .Input(x, data_format)
                .Output(y, data_format)
                .Attr("ksize", kernel_64)
                .Attr("strides", stride_64)
                .Attr("padding_mode", std::string("CALCULATED"))
                .Attr("pads", pad_64)
                .Attr("data_format", data_format)
                .Attr("global_pooling", false)
                .Attr("ceil_mode", ceil_mode)
                .Attr("exclusive", true)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
      npu_command.Run();
      OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
      //PrintResult(y);
      //std::cout<<"AvgPool Execute Over"<<std::endl; 
  };
};
#define REGISTER_AVG_POOLING_KERNELS(dtype)                                           \
    REGISTER_USER_KERNEL("avgpool_2d")                                                \
    .SetCreateFn<AvgPool2dNpuKernel>()                                                \
    .SetIsMatchedHob((user_op::HobDeviceType() == kNPU)                               \
                    && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));  \

REGISTER_AVG_POOLING_KERNELS(float);
REGISTER_AVG_POOLING_KERNELS(float16);

template<typename T>
class AvgPool2dGradNpuKernel final : public user_op::OpKernel {
 public:
  AvgPool2dGradNpuKernel() = default;
  ~AvgPool2dGradNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    std::string data_format = ctx->Attr<std::string>("data_format");
    std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
    std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("stride");
    std::vector<int64_t> kernel_64 = {1, 1, kernel_size[0], kernel_size[1]};
    std::vector<int64_t> stride_64 = {1, 1, stride[0], stride[1]};
    std::vector<int64_t> pad_64 = {padding[0], padding[0], padding[1], padding[1]};
    bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    bool count_include_pad = ctx->Attr<bool>("count_include_pad");// dck_caution_here
    int32_t divisor_override = ctx->Attr<int32_t>("divisor_override");

    std::vector<int32_t> dx_shape;
    for(size_t i=0; i<dx->shape().NumAxes();++i)
    {
      dx_shape.push_back(dx->shape().ptr()[i]);
    }
    VECTOR_PRINT(dx_shape);
    void* tensor_ptr = nullptr;
    std::vector<int64_t> shape_desc;
    shape_desc.push_back(dx_shape.size());
    AclTensorWrapper wrap(tensor_ptr,
                      ACL_INT32,
                      shape_desc.size(),
                      shape_desc.data(),
                      ACL_FORMAT_ND,
                      mulVector(shape_desc)*sizeof(int32_t),
                      dx_shape.data(),
                      /*isConst*/true);    
    NpuCommand npu_command;
    npu_command.OpName("AvgPoolV2Grad")
               .Input(wrap)
               .Input(dy,"channel_nd")
               .Output(dx,"channel_nd")
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
    //PrintResult(dx);
    //std::cout<<"AvgPoolGrad Execute Over"<<std::endl; 
  };
};
#define REGISTER_AVG_POOLING_NPU_KERNELS(dtype)                                         \
       REGISTER_USER_KERNEL("avgpool_2d_grad")                                           \
       .SetCreateFn<AvgPool2dGradNpuKernel<dtype>>()                                            \
       .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                   \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));  
                       
REGISTER_AVG_POOLING_NPU_KERNELS(float16)
REGISTER_AVG_POOLING_NPU_KERNELS(float)
} // namespace oneflow
