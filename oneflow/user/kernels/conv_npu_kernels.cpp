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

#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#define GetCurrentDirPath getcwd
#define Mkdir(path, mode) mkdir(path, mode)

using namespace std;
namespace oneflow {
namespace {

const size_t kMaxPathLen = 4096U;
std::string GetCurDirPath() {
  char buff[kMaxPathLen] = {'\0'};
  GetCurrentDirPath(buff, kMaxPathLen);
  return std::string(buff);
}
void MakeCompileCacheDirAndSetOption() {
  auto compile_cache_dir = GetCurDirPath() + "/cache";
  // mode : 750
  auto ret = Mkdir(compile_cache_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
  if (ret == -1) {
    if (errno != EEXIST) {
      std::cout<<"make compile cache directory error: "<<strerror(errno)<<std::endl;
      return;
    }
  }
  std::string val = "enable";
  std::cout<<val<<" "<<compile_cache_dir<<std::endl;
  OF_NPU_CHECK(aclSetCompileopt(aclCompileOpt::ACL_OP_COMPILER_CACHE_MODE, val.c_str()));
  OF_NPU_CHECK(aclSetCompileopt(aclCompileOpt::ACL_OP_COMPILER_CACHE_DIR, compile_cache_dir.c_str()));
  std::cout<<"MakeCompileCacheDirAndSetOption Over"<<std::endl;
}

template<typename T, size_t NDims>
class ConvNpuKernel final : public user_op::OpKernel {
 public:
  ConvNpuKernel() = default;
  ~ConvNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    // MakeCompileCacheDirAndSetOption();
    // user_op::Tensor : /home/HDD/dck/oneflow/oneflow/core/framework/user_op_tensor.h
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    //user_op::Tensor* buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    //const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding_before");
    std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");
    std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    std::string data_format = ctx->Attr<std::string>("data_format");
    std::vector<int64_t> strides_64 = {1, 1, stride[0], stride[1]};
    std::vector<int64_t> paddings_64 = { padding[0], padding[0], padding[1], padding[1]};
    std::vector<int64_t> dilations_64 = {1, 1, dilation[0], dilation[1]};
    NpuCommand npu_command;
    npu_command.OpName("Conv2D")
              .Input(in, data_format)
              .Input(weight, data_format)
              .Output(out, data_format)
              .Attr("strides", strides_64)
              .Attr("pads", paddings_64)
              .Attr("dilations", dilations_64)
              .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
              .Check();
    npu_command.Run()
               .Realease();
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
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding_before");
    std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");
    std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    std::string data_format = ctx->Attr<std::string>("data_format");

    std::vector<int64_t> strides_64 = {1, 1, stride[0], stride[1]};
    std::vector<int64_t> paddings_64 = { padding[0], padding[0], padding[1], padding[1]};
    std::vector<int64_t> dilations_64 = {1, 1, dilation[0], dilation[1]};
    
    std::vector<int> x_shape;
    for(size_t i=0; i<dx->shape_view().NumAxes();++i)
    {
      x_shape.push_back(dx->shape_view().ptr()[i]);
    }    
    std::vector<int64_t> shape_desc;
    shape_desc.push_back(x_shape.size());
    CHECK_EQ(tmp_buffer->shape_view().elem_cnt(), mulVector(shape_desc)*sizeof(int32_t));
    std::string key = "ConvDataGradNpu" + ShapeToString(x_shape);
    if(!const_tensor_map.count(key))  const_tensor_map[key] = x_shape;
    if(!shape_map.count(key)) shape_map[key] = shape_desc;
    // AclTensorWrapper wrap(tmp_buffer->mut_dptr<void>(),
    //                   ACL_INT32,
    //                   shape_desc.size(),
    //                   shape_desc.data(),
    //                   ACL_FORMAT_ND,
    //                   mulVector(shape_desc)*sizeof(int32_t),
    //                   x_shape.data(),
    //                   key);
    int64_t groups = 1;
    NpuCommand npu_command;
    npu_command.OpName("Conv2DBackpropInput")
               .Input(key, x_shape.size(), ACL_INT32)
               .Input(filter, data_format, "filter")
               .Input(dy, data_format, "out_backprop")
               .Output(dx, data_format, "y")
               .Attr("strides", strides_64)
               .Attr("pads", paddings_64)
               .Attr("dilations", dilations_64)
               .Attr("groups", groups)
               .Attr("data_format", data_format)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
    //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    //PrintResult(dx);
    //std::cout<<"ConvDataGrad Execute Over"<<std::endl; 

  }
};

#define REGISTER_CONV_DATA_GRAD_KERNEL(op_name, dtype)                                     \
  REGISTER_USER_KERNEL(#op_name)                                                           \
      .SetCreateFn<ConvDataGradNpuKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                      \
                       && (user_op::HobAttr<int32_t>("groups") == 1)                       \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
        size_t tmp_buffer_size = 0;                                                        \
        const auto& x_like_shape = ctx->InputTensorDesc("x_like", 0).shape();                \
        tmp_buffer_size += x_like_shape.NumAxes()*sizeof(int);                                          \
        return tmp_buffer_size;                                                              \
      })

REGISTER_CONV_DATA_GRAD_KERNEL(conv_data_grad, float);
REGISTER_CONV_DATA_GRAD_KERNEL(conv_data_grad, float16);

template<typename T>
class ConvFilterGradNpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradNpuKernel);
  ConvFilterGradNpuKernel() = default;
  ~ConvFilterGradNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }


 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {           
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* filter_diff = ctx->Tensor4ArgNameAndIndex("filter_diff", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    //user_op::Tensor* col_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding_before");
    std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");
    std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    std::string data_format = ctx->Attr<std::string>("data_format");

    std::vector<int64_t> strides_64 = {1, 1, stride[0], stride[1]};
    std::vector<int64_t> paddings_64 = { padding[0], padding[0], padding[1], padding[1]};
    std::vector<int64_t> dilations_64 = {1, 1, dilation[0], dilation[1]};
    std::vector<int> filter_shape;
    for(size_t i=0; i<filter_diff->shape_view().NumAxes();++i)
    {
      filter_shape.push_back(filter_diff->shape_view().ptr()[i]);
    }
    
    std::vector<int64_t> shape_desc;
    shape_desc.push_back(filter_shape.size());
    CHECK_EQ(tmp_buffer->shape_view().elem_cnt(), mulVector(shape_desc)*sizeof(int32_t));
    std::string key = "ConvDataFilterNpu" + ShapeToString(filter_shape);
    if(!const_tensor_map.count(key)) const_tensor_map[key] = filter_shape;
    if(!shape_map.count(key)) shape_map[key] = shape_desc;
    // AclTensorWrapper wrap(tmp_buffer->mut_dptr<void>(),
    //                   ACL_INT32,
    //                   shape_desc.size(),
    //                   shape_desc.data(),
    //                   ACL_FORMAT_ND,
    //                   mulVector(shape_desc)*sizeof(int32_t),
    //                   filter_shape.data(),
    //                   key);
    int64_t groups = 1;

    NpuCommand npu_command;
    npu_command.OpName("Conv2DBackpropFilter")
               .Input(x, data_format, "x")
               .Input(key, filter_shape.size(), ACL_INT32)
               .Input(dy, data_format, "out_backprop")
               .Output(filter_diff, data_format, "y")
               .Attr("strides", strides_64)
               .Attr("pads", paddings_64)
               .Attr("dilations", dilations_64)
               .Attr("groups", groups)
               .Attr("data_format", data_format)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
    //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    //PrintResult(filter_diff);
    //std::cout<<"ConvFilterGrad Execute Over"<<std::endl; 
  }
};
#define REGISTER_CONV_FILTER_GRAD_KERNEL(op_name, dtype)                                        \
  REGISTER_USER_KERNEL(#op_name)                                                                \
      .SetCreateFn<ConvFilterGradNpuKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                           \
                       && (user_op::HobAttr<int32_t>("groups") == 1)                            \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))         \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
        size_t tmp_buffer_size = 0;                                                        \
        const auto& filter_diff_shape = ctx->InputTensorDesc("filter_diff", 0).shape();                \
        tmp_buffer_size += filter_diff_shape.NumAxes() * sizeof(int);                                          \
        return tmp_buffer_size;                                                              \
      })

REGISTER_CONV_FILTER_GRAD_KERNEL(conv_filter_grad, float);
REGISTER_CONV_FILTER_GRAD_KERNEL(conv_filter_grad, double);
REGISTER_CONV_FILTER_GRAD_KERNEL(conv_filter_grad, float16);
} // namespace
} // namespace oneflow

#endif