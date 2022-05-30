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
#include "oneflow/user/kernels/pooling_kernel_util.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

int64_t CeilDiv(int64_t value, int64_t factor) {
  int64_t value_num = 0;
  if (factor == 0) {
    return value_num;
  }
  if (value % factor == 0) {
    value_num = value / factor;
  } else {
    value_num = value / factor + 1;
  }

  return value_num;
}
template<typename T>
class MaxPool2dNpuKernel final : public user_op::OpKernel {
 public:
  MaxPool2dNpuKernel() = default;
  ~MaxPool2dNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {

    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    std::vector<int32_t> ksize = ctx->Attr<std::vector<int32_t>>("kernel_size");
    std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
    std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("stride");
    std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation");
    bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    std::vector<int64_t> ksize_64 = {1,  ksize[0], ksize[1], 1};
    std::vector<int64_t> strides_64 = {1, stride[0], stride[1], 1};
    std::vector<int64_t> paddings_64 = { 1, padding[0], padding[1], 1};
    std::vector<int64_t> dilations_64 = {1, dilation[0], dilation[1], 1};

    MaxPoolTensorWrapper indice_wrap( indice->mut_dptr<void>(), ACL_UINT16, ACL_FORMAT_NCHW, ACL_FORMAT_NC1HWC0,
                                        indice->shape().NumAxes(), indice->shape().ptr(), 
                                        indice->shape().elem_cnt()*GetSizeOfDataType(indice->data_type()));
    std::string data_format = ctx->Attr<std::string>("data_format");
    
    NpuCommand npu_command;
    npu_command.OpName("MaxPoolWithArgmaxV1")
               .Input(x,data_format)
               .Output(y,data_format)
               .Output(indice_wrap)
               .Attr("ksize", ksize_64)
               .Attr("strides", strides_64)
               .Attr("pads", paddings_64)
               .Attr("dilation", dilations_64)
               .Attr("ceil_mode", ceil_mode)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    //PrintResult(y);
    
  };
};

template<typename T>
class MaxPool2dGradNpuKernel final : public user_op::OpKernel {
 public:
  MaxPool2dGradNpuKernel() = default;
  ~MaxPool2dGradNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);

    //user_op::TensorDesc* = const_cast<user_op::TensorDesc*>(indice_desc_c);
    std::vector<int32_t> ksize = ctx->Attr<std::vector<int32_t>>("kernel_size");
    std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
    std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("stride");
    std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation");
    bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    std::vector<int64_t> ksize_64 = {1,  ksize[0], ksize[1], 1};
    std::vector<int64_t> strides_64 = {1, stride[0], stride[1], 1};
    std::vector<int64_t> paddings_64 = { 1, padding[0], padding[1], 1};
    std::vector<int64_t> dilations_64 = {1, dilation[0], dilation[1], 1};
    std::string data_format = ctx->Attr<std::string>("data_format");
    
    // MaxPoolTensorWrapper wrap(indice->mut_dptr<void>(), ACL_UINT16, ACL_FORMAT_NCHW, ACL_FORMAT_NC1HWC0,
    //                             indice_dim.size(), indice_dim.data(), 
    //                             mulVector(indice_dim)*8);
    MaxPoolTensorWrapper wrap( indice->mut_dptr<void>(), ACL_UINT16, ACL_FORMAT_NCHW, ACL_FORMAT_NC1HWC0,
                                indice->shape().NumAxes(), indice->shape().ptr(), 
                                indice->shape().elem_cnt()*GetSizeOfDataType(indice->data_type()));
    NpuCommand npu_command;
    npu_command.OpName("MaxPoolGradWithArgmaxV1")
               .Input(x,data_format)
               .Input(dy,data_format)
               .Input(wrap)
               .Output(dx,data_format)
               .Attr("ksize", ksize_64)
               .Attr("strides", strides_64)
               .Attr("pads", paddings_64)
               .Attr("dilations", dilations_64)
               .Attr("ceil_mode", ceil_mode)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    //PrintResult(dx);
    //std::cout<<"MaxPoolGrad Execute Over"<<std::endl; 
  };

};
#define REGISTER_POOLING_NPU_KERNELS(dtype)                                             \
  REGISTER_USER_KERNEL("maxpool_2d")                                                    \
      .SetCreateFn<MaxPool2dNpuKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                   \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_2d_grad")                                               \
      .SetCreateFn<MaxPool2dGradNpuKernel<dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                   \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); 

REGISTER_POOLING_NPU_KERNELS(float)
REGISTER_POOLING_NPU_KERNELS(float16)
REGISTER_POOLING_NPU_KERNELS(double)                    
} // namespace oneflow
