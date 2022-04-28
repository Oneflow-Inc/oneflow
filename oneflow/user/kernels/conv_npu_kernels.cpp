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
#ifdef WITH_NPU
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/nn_util.h"
//#include "oneflow/core/device/cudnn_conv_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
//#include "oneflow/core/kernel/cuda_graph_support.h"
#include "acl/ops/acl_cblas.h"
#include "oneflow/user/ops/npu_command.h"

using namespace std;
namespace oneflow {
namespace {

#define VECTOR_PRINT(x) std::cout<<#x<<" ";\
                        for(auto& i:x) { std::cout<<i<<" ";}\
                        std::cout<<std::endl;

template<typename T, size_t NDims>
class ConvNpuKernel final : public user_op::OpKernel {
 public:
  ConvNpuKernel() = default;
  ~ConvNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    
    // user_op::Tensor : /home/HDD/dck/oneflow/oneflow/core/framework/user_op_tensor.h
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    //user_op::Tensor* buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::TensorDesc* in_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);

    const user_op::TensorDesc* weight_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("weight", 0);
    const user_op::TensorDesc* out_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);

    //const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding_before");
    std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");
    std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");

    std::vector<int64_t> strides_64 = {1, 1, stride[0], stride[1]};
    std::vector<int64_t> paddings_64 = { padding[0], padding[0], padding[1], padding[1]};
    std::vector<int64_t> dilations_64 = {1, 1, dilation[0], dilation[1]};
    VECTOR_PRINT(strides_64);
    VECTOR_PRINT(paddings_64);
    VECTOR_PRINT(dilations_64);    
    NpuCommand npu_command;
    npu_command.OpName("Conv2D")
              .Input(in, "channel_last")
              .Input(weight, "channel_first")
              .Output(out, "channel_last")
              //.InputDesc(in_tensor_desc, "channel_last")
              //.InputDesc(weight_tensor_desc, "channel_first")
              //.OutputDesc(out_tensor_desc, "channel_last")
              .Attr("strides", strides_64)
              .Attr("pads", paddings_64)
              .Attr("dilations", dilations_64)
              .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
              .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    PrintResult(out->mut_dptr<void>(),out->shape().elem_cnt() * GetSizeOfDataType(out->data_type()));
    std::cout<<"Execute Over"<<std::endl; 
  };
};
#define REGISTER_CONV_KERNEL(op_name, dtype, ndims)                                                \
  REGISTER_USER_KERNEL(#op_name)                                                                   \
      .SetCreateFn<ConvNpuKernel<dtype, ndims>>()                                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                              \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))            

REGISTER_CONV_KERNEL(conv1d, float, 1);
REGISTER_CONV_KERNEL(conv2d, float, 2);
REGISTER_CONV_KERNEL(conv3d, float, 3);
REGISTER_CONV_KERNEL(conv1d, double, 1);
REGISTER_CONV_KERNEL(conv2d, double, 2);
REGISTER_CONV_KERNEL(conv3d, double, 3);
REGISTER_CONV_KERNEL(conv1d, float16, 1);
REGISTER_CONV_KERNEL(conv2d, float16, 2);
REGISTER_CONV_KERNEL(conv3d, float16, 3);

} // namespace
} // namespace oneflow

#endif