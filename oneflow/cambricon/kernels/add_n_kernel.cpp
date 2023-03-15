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
#include <cstdint>
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/ep/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

template<typename T>
class MluAddNKernel final : public user_op::OpKernel {
 public:
  MluAddNKernel() = default;
  ~MluAddNKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in_0 = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType data_type = out->data_type();
    const size_t count = out->shape_view().elem_cnt();

    size_t in_num = ctx->inputs().size();
    std::vector<const void*> input_dptrs_vec(in_num);
    for (size_t i = 0; i < in_num; ++i) {
      const auto* in_i = ctx->Tensor4ArgNameAndIndex("in", i);
      CHECK_EQ(in_i->shape_view().elem_cnt(), count);
      CHECK_EQ(in_i->data_type(), data_type);
      input_dptrs_vec[i] = in_i->dptr();
    }
    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set(in_0);
    output_desc.set(out);
    std::vector<cnnlTensorDescriptor_t> input_descs_vec{in_num, input_desc.desc()};
    size_t addn_workspace_size = 0;
    void* addn_workspace = nullptr;

    OF_CNNL_CHECK(cnnlAddN_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                           input_descs_vec.data(), input_dptrs_vec.data(), in_num,
                           output_desc.desc(), out->mut_dptr(), addn_workspace, addn_workspace_size));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADDN_MLU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("add_n").SetCreateFn<MluAddNKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                 \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_ADDN_MLU_KERNEL(float)
REGISTER_ADDN_MLU_KERNEL(float16)
REGISTER_ADDN_MLU_KERNEL(int8_t)
REGISTER_ADDN_MLU_KERNEL(uint8_t)
REGISTER_ADDN_MLU_KERNEL(int32_t)

}  // namespace oneflow
