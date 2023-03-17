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
#include "oneflow/core/common/scalar.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

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
    // TODO: if(input->shape_view().elem_cnt() == 0)

    cnnlReduceDescriptor_t reduce_desc;
    OF_CNNL_CHECK(cnnlCreateReduceDescriptor(&reduce_desc));
    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set(input);
    output_desc.set(output);

    int axis_num = axis.size();
    int reduce_axis[axis_num];
    for (int i = 0; i < axis_num; ++i) { reduce_axis[i] = axis.at(i); }
    auto input_dtype = ConvertToCnnlDataType(input->data_type());
    OF_CNNL_CHECK(cnnlSetReduceDescriptor(reduce_desc, reduce_axis, axis_num, CNNL_REDUCE_ADD,
                                          input_dtype, CNNL_NOT_PROPAGATE_NAN,
                                          CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES));

    OF_CNNL_CHECK(cnnlReduce(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), reduce_desc,
                             nullptr, 0, nullptr, input_desc.desc(), input->dptr(), 0, nullptr,
                             nullptr, output_desc.desc(), output->mut_dptr()));

    OF_CNNL_CHECK(cnnlDestroyReduceDescriptor(reduce_desc));
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

}  // namespace oneflow
