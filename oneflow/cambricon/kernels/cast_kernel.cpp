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
#include "cnnl.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/primitive/cast.h"

#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T>
class MluCastKernel final : public user_op::OpKernel {
 public:
  MluCastKernel() = default;
  ~MluCastKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType in_data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
    const DataType out_data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
    CnnlTensorDescriptor in_desc(in), out_decs(out);

    if ((in_data_type == kUInt8 && out_data_type == kBool)
        || (in_data_type == kBool && out_data_type == kInt64)) {
      size_t tmp_out_workspace_size = out->shape_view().elem_cnt() * sizeof(kInt32);
      CnnlWorkspace tmp_out_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                           tmp_out_workspace_size);
      void* tmp_out_ptr = tmp_out_cnnl_workspace.dptr();
      CnnlTensorDescriptor tmp_out_desc;
      tmp_out_desc.set(out, ConvertToCnnlDataType(kInt32));
      if (in_data_type == kUInt8) {
        OF_CNNL_CHECK(cnnlCastDataType(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                       in_desc.desc(), in->dptr(), CNNL_CAST_UINT8_TO_INT32,
                                       tmp_out_desc.desc(), tmp_out_ptr));
      } else {
        OF_CNNL_CHECK(cnnlCastDataType(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                       in_desc.desc(), in->dptr(), CNNL_CAST_BOOL_TO_INT32,
                                       tmp_out_desc.desc(), tmp_out_ptr));
      }
      if (out_data_type == kBool) {
        OF_CNNL_CHECK(cnnlCastDataType(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                       tmp_out_desc.desc(), tmp_out_ptr, CNNL_CAST_INT32_TO_BOOL,
                                       out_decs.desc(), out->mut_dptr()));
      } else {
        OF_CNNL_CHECK(cnnlCastDataType(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                       tmp_out_desc.desc(), tmp_out_ptr, CNNL_CAST_INT32_TO_INT64,
                                       out_decs.desc(), out->mut_dptr()));
      }
      return;
    }

    cnnlCastDataType_t type = ep::primitive::GetCnnlCastType(in_data_type, out_data_type);

    // primitive cast does not support non-contiguous, so we implement another one here.
    OF_CNNL_CHECK(cnnlCastDataType(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                   in_desc.desc(), in->dptr(), type, out_decs.desc(),
                                   out->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CAST_MLU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("cast").SetCreateFn<MluCastKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_CAST_MLU_KERNEL(bool)
REGISTER_CAST_MLU_KERNEL(float)
REGISTER_CAST_MLU_KERNEL(float16)
REGISTER_CAST_MLU_KERNEL(int8_t)
REGISTER_CAST_MLU_KERNEL(uint8_t)
REGISTER_CAST_MLU_KERNEL(int32_t)
REGISTER_CAST_MLU_KERNEL(int64_t)

}  // namespace oneflow
