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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace {

template<DeviceType device_type, typename T, typename K>
class ReduceKernel final : public user_op::OpKernel {
 public:
  ReduceKernel() = default;
  ~ReduceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");

    auto cnnl_dtype = ConvertToCnnlDataType(output->data_type());

    CnnlReduceDescriptor reduce_desc;
    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set_reduce(input);

    auto reduce_mode = CNNL_REDUCE_ADD;
    auto reduce_indices = CNNL_REDUCE_NO_INDICES;
    auto reduce_indices_type = CNNL_32BIT_INDICES;

    if (axis.size() == input->shape_view().NumAxes()) {
      std::vector<int32_t> full_reduce(1, -1);
      std::vector<int32_t> fake_size(input->shape_view().NumAxes(), 1);
      reduce_desc.set(cnnl_dtype, full_reduce, reduce_mode, reduce_indices, reduce_indices_type);
      output_desc.set(fake_size.size(), fake_size.data(), cnnl_dtype, CNNL_LAYOUT_NCHW);
    } else {
      reduce_desc.set(cnnl_dtype, axis, reduce_mode, reduce_indices, reduce_indices_type);
      output_desc.set(output);
    }
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                               input_desc.desc(), output_desc.desc(),
                                               reduce_desc.mut_desc(), &workspace_size));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);

    OF_CNNL_CHECK(cnnlReduce(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), reduce_desc.desc(),
                             workspace.dptr(), workspace_size, nullptr, input_desc.desc(),
                             input->dptr(), 0, nullptr, nullptr, output_desc.desc(),
                             output->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REDUCE_MLU_KERNEL(op_name, device, dtype)                                         \
  REGISTER_USER_KERNEL(op_name).SetCreateFn<ReduceKernel<device, dtype, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                                         \
      && (user_op::HobDataType("output_tensor", 0) == GetDataType<dtype>::value));

#define REGISTER_REDUCE_SUM_KERNELS(device, dtype) \
  REGISTER_REDUCE_MLU_KERNEL("reduce_sum", device, dtype)

#define REGISTER_REDUCE_SUM_KERNELS_BY_DEVICE(device) \
  REGISTER_REDUCE_SUM_KERNELS(device, float)          \
  REGISTER_REDUCE_SUM_KERNELS(device, float16)        \
  REGISTER_REDUCE_SUM_KERNELS(device, int32_t)

REGISTER_REDUCE_SUM_KERNELS_BY_DEVICE(DeviceType::kMLU)

}  // namespace
}  // namespace oneflow
