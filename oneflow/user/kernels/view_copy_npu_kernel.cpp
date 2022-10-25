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
class ViewCopyNpuKernel final : public user_op::OpKernel {
 public:
  ViewCopyNpuKernel() = default;
  ~ViewCopyNpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& in_shape = in->shape_view();
    const ShapeView& out_shape = out->shape_view();
    CHECK_EQ(out->shape_view(), in_shape);
    const DataType in_data_type = in->data_type();
    CHECK_EQ(out->data_type(), in_data_type);

    std::vector<int> in_stride(in->stride().begin(), in->stride().end());
    for(int& is:in_stride){
        is = 0;
    }
    std::vector<int64_t> in_stride_desc = {static_cast<int>(in_stride.size())};
    HostTensorWrapper in_stride_wrap(ACL_INT32, ACL_FORMAT_ND, in_stride_desc.size(), in_stride_desc.data(),
                            in_stride.size()*sizeof(int), in_stride.data());

    std::vector<int> out_stride(out->stride().begin(), out->stride().end());
    std::vector<int64_t> out_stride_desc = {static_cast<int>(out_stride.size())};
    HostTensorWrapper out_stride_wrap(ACL_INT32, ACL_FORMAT_ND, out_stride_desc.size(), out_stride_desc.data(),
                            out_stride.size()*sizeof(int), out_stride.data());

    DimVector out_shape_dim_v;
    out_shape.ToDimVector(&out_shape_dim_v);
    std::vector<int> out_shape_vector;
    for(auto sh:out_shape_dim_v)
    {
        out_shape_vector.push_back(sh);
    }
    std::vector<int64_t> out_shape_desc = {static_cast<int>(out_shape_vector.size())};
    HostTensorWrapper out_shape_wrap(ACL_INT32, ACL_FORMAT_ND, out_shape_desc.size(), out_shape_desc.data(),
                            out_shape_vector.size()*sizeof(int), out_shape_vector.data());  

    DimVector in_shape_dim_v;
    in_shape.ToDimVector(&in_shape_dim_v);
    std::vector<int> in_shape_vector;
    for(auto sh:in_shape_dim_v)
    {
        in_shape_vector.push_back(sh);
    }
    std::vector<int64_t> in_shape_desc = {static_cast<int>(in_shape_vector.size())};
    HostTensorWrapper in_shape_wrap(ACL_INT32, ACL_FORMAT_ND, in_shape_desc.size(), in_shape_desc.data(),
                            in_shape_vector.size()*sizeof(int), in_shape_vector.data());  

    std::vector<int64_t> storage_offset_v = {0};
    std::vector<int64_t> offset_desc = {static_cast<int>(storage_offset_v.size())};
 
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));
    AclTensorWrapper offset_wrap(tmp_buffer->mut_dptr<void>(), ACL_INT64, offset_desc.size(), offset_desc.data(), ACL_FORMAT_ND,
                            sizeof(int64_t), storage_offset_v.data());
    std::vector<int64_t> RealShape = {out_shape_vector.begin(), out_shape_vector.end()};
    for(int i=1;i<RealShape.size();++i){
        RealShape[i] = std::max(RealShape[i], out->stride().at(i-1));
    }
    NpuCommand npu_command;
    npu_command.OpName("ViewCopy")
               .InputWithShape(out, RealShape)
               .Input(out_shape_wrap)
               .Input(out_stride_wrap)
               .Input(offset_wrap)
               .Input(in)
               .Input(in_shape_wrap)
               .Input(in_stride_wrap)
               .Input(offset_wrap)
               .Output(out)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));  
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_VIEW_COPY_NPU_KERNEL(T)            \
  REGISTER_USER_KERNEL("view_copy_npu")                          \
      .SetCreateFn<ViewCopyNpuKernel<T>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU) \
                       && (user_op::HobDataType("in", 0) == GetDataType<T>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        return sizeof(int64_t); \
      })
REGISTER_VIEW_COPY_NPU_KERNEL(float);
REGISTER_VIEW_COPY_NPU_KERNEL(float16);

}  // namespace
}  // namespace oneflow
