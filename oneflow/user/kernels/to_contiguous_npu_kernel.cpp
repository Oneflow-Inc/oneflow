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
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/to_contiguous_kernel.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/user/ops/npu_command.h"

namespace oneflow {

namespace {

template<typename T>
class ToContiguousNpuKernel final : public user_op::OpKernel {
 public:
  ToContiguousNpuKernel() = default;
  ~ToContiguousNpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape_view();
    const ShapeView& out_shape = out->shape_view();
    CHECK_EQ(out->shape_view(), in_shape);
    const DataType in_data_type = in->data_type();
    CHECK_EQ(out->data_type(), in_data_type);

    std::vector<int> in_stride(in->stride().begin(), in->stride().end());
    std::vector<int64_t> stride_desc = {static_cast<int>(in_stride.size())};
    HostTensorWrapper stride_wrap(ACL_INT32, ACL_FORMAT_ND, stride_desc.size(), stride_desc.data(),
                            in_stride.size()*sizeof(int), in_stride.data());
    DimVector shape_dim_v;
    out_shape.ToDimVector(&shape_dim_v);
    std::vector<int> shape_vector;
    for(auto sh:shape_dim_v)
    {
        shape_vector.push_back(sh);
    }
    std::vector<int64_t> shape_desc = {static_cast<int>(shape_vector.size())};
    HostTensorWrapper shape_wrap(ACL_INT32, ACL_FORMAT_ND, shape_desc.size(), shape_desc.data(),
                            shape_vector.size()*sizeof(int), shape_vector.data());    

    std::vector<int> storage_offset_v = {0};
    std::vector<int64_t> offset_desc = {static_cast<int>(storage_offset_v.size())};
    HostTensorWrapper offset_wrap(ACL_INT32, ACL_FORMAT_ND, offset_desc.size(), offset_desc.data(),
                            storage_offset_v.size()*sizeof(int), storage_offset_v.data());  
    
    std::vector<int64_t> RealShape(shape_vector.begin(), shape_vector.end());
    for(int i=1;i<shape_vector.size();++i)
    {
      RealShape[i] = std::max(RealShape[i], static_cast<int64_t>(in_stride[i-1]));
    }
    // std::cout<<"ToContiguous "<<ShapeToString(RealShape)<<std::endl;
    NpuCommand npu_command;
    npu_command.OpName("AsStrided")
               .InputWithShape(in, RealShape)
               .Input(shape_wrap)
               .Input(stride_wrap)
               .Input(offset_wrap)
               .Output(out)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TO_CONTIGUOUS_NPU_KERNEL(T)            \
  REGISTER_USER_KERNEL("to_contiguous")                          \
      .SetCreateFn<ToContiguousNpuKernel<T>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU) \
                       && (user_op::HobDataType("in", 0) == GetDataType<T>::value));

REGISTER_TO_CONTIGUOUS_NPU_KERNEL(float);
REGISTER_TO_CONTIGUOUS_NPU_KERNEL(float16);

}  // namespace
}  // namespace oneflow
