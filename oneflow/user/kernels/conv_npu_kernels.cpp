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
    PrintResult(out);
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


template<typename T>
class ConvDataGradNpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvDataGradNpuKernel);
  ConvDataGradNpuKernel() = default;
  ~ConvDataGradNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }


 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* x_like = ctx->Tensor4ArgNameAndIndex("x_like", 0);
    user_op::Tensor* filter = ctx->Tensor4ArgNameAndIndex("filter", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    //user_op::Tensor* col_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    Memset<DeviceType::kNPU>(ctx->stream(), dx->mut_dptr<T>(), 0,
                             dx->shape().elem_cnt() * sizeof(T));

    std::cout<<"ConvNpuDataGradCpuKernel"<<std::endl;
    // if (ctx->has_input("_add_to_output", 0)) {
    //   const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
    //   CHECK_EQ(add_to_output->data_type(), dx->data_type());
    //   CHECK_EQ(add_to_output->shape(), dx->shape());
    //   std::unique_ptr<ep::primitive::Add> primitive =
    //       ep::primitive::NewPrimitive<ep::primitive::AddFactory>(DeviceType::kCPU,
    //                                                              add_to_output->data_type());
    //   CHECK(primitive);
    //   primitive->Launch(ctx->stream(), dx->dptr<T>(), add_to_output->dptr<T>(), dx->mut_dptr<T>(),
    //                     add_to_output->shape().elem_cnt());
    // }
  }
};

#define REGISTER_CONV_DATA_GRAD_KERNEL(op_name, dtype)                                     \
  REGISTER_USER_KERNEL(#op_name)                                                           \
      .SetCreateFn<ConvDataGradNpuKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                      \
                       && (user_op::HobAttr<int32_t>("groups") == 1)                       \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))    \
      // .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
      //   size_t tmp_buffer_size = 0;                                                        \
      //   const auto& out_diff_shape = ctx->InputTensorDesc("dy", 0).shape();                \
      //   const auto& weight_shape = ctx->InputTensorDesc("filter", 0).shape();              \
      //                                                                                      \
      //   int64_t idx_offset = IdxOffset(ctx->Attr<std::string>("data_format"));             \
      //   tmp_buffer_size +=                                                                 \
      //       CalcElemNumOfColBuf(out_diff_shape, weight_shape, idx_offset) * sizeof(dtype); \
      //   return tmp_buffer_size;                                                            \
      // })

REGISTER_CONV_DATA_GRAD_KERNEL(conv_data_grad, float);
REGISTER_CONV_DATA_GRAD_KERNEL(conv_data_grad, float16);
} // namespace
} // namespace oneflow

#endif