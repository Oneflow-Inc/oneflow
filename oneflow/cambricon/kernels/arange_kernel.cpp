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
#include "oneflow/cambricon/cnnl/cnnl_types.h"
#include "oneflow/cambricon/ep/primitive/cast.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

template<typename T>
class MluArangeKernel final : public user_op::OpKernel {
 public:
  MluArangeKernel() = default;
  ~MluArangeKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();
    const DataType dtype = ctx->Attr<DataType>("dtype");
    int32_t start_int = 0;
    int32_t step_int = 0;
    float start_float = 0.0;
    float step_float = 0.0;
    CnnlTensorDescriptor tmp_out_desc, out_decs;
    out_decs.set(out);

    DataType tmp_out_data_type;
    size_t tmp_out_workspace_size = out->shape_view().elem_cnt();
    if constexpr (std::is_same_v<T, float>) {
      tmp_out_workspace_size *= GetSizeOfDataType(kFloat);
      tmp_out_data_type = kFloat;
    } else if constexpr (std::is_same_v<T, float16>) {
      tmp_out_workspace_size *= GetSizeOfDataType(kFloat16);
      tmp_out_data_type = kFloat16;
    } else {
      tmp_out_workspace_size *= GetSizeOfDataType(kInt32);
      tmp_out_data_type = kInt32;
    }
    CnnlWorkspace tmp_out_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                         tmp_out_workspace_size);
    void* tmp_out_ptr = tmp_out_cnnl_workspace.dptr();

    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float16>) {
      tmp_out_desc.set(out, ConvertToCnnlDataType(GetDataType<T>::value));
      start_float = static_cast<float>(ctx->Attr<double>("float_start"));
      step_float = static_cast<float>(ctx->Attr<double>("float_delta"));
      OF_CNNL_CHECK(cnnlArange_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                  CNNL_COMPUTATION_HIGH_PRECISION, (void*)&start_float,
                                  (void*)&step_float, tmp_out_desc.desc(), tmp_out_ptr));
    } else {
      tmp_out_desc.set(out, ConvertToCnnlDataType(kInt32));
      start_int = static_cast<int32_t>(ctx->Attr<int64_t>("integer_start"));
      step_int = static_cast<int32_t>(ctx->Attr<int64_t>("integer_delta"));
      OF_CNNL_CHECK(cnnlArange_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                  CNNL_COMPUTATION_HIGH_PRECISION, (void*)&start_int,
                                  (void*)&step_int, tmp_out_desc.desc(), tmp_out_ptr));
    }

    if (tmp_out_data_type != dtype) {
      cnnlCastDataType_t type = ep::primitive::GetCnnlCastType(tmp_out_data_type, dtype);

      OF_CNNL_CHECK(cnnlCastDataType(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                     tmp_out_desc.desc(), tmp_out_ptr, type, out_decs.desc(),
                                     output));
    } else {
      OF_MLU_CHECK(cnrtMemcpyAsync(
          output, tmp_out_ptr, out->shape_view().elem_cnt() * GetSizeOfDataType(dtype),
          ctx->stream()->As<ep::MluStream>()->mlu_stream(), cnrtMemcpyDevToDev));
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ARANGE_MLU_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("arange").SetCreateFn<MluArangeKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                    \
      && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

REGISTER_ARANGE_MLU_KERNEL(float)
REGISTER_ARANGE_MLU_KERNEL(float16)
REGISTER_ARANGE_MLU_KERNEL(int8_t)
REGISTER_ARANGE_MLU_KERNEL(uint8_t)
REGISTER_ARANGE_MLU_KERNEL(int32_t)
REGISTER_ARANGE_MLU_KERNEL(uint32_t)
REGISTER_ARANGE_MLU_KERNEL(int64_t)
REGISTER_ARANGE_MLU_KERNEL(uint64_t)

}  // namespace oneflow
