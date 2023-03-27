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
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace user_op {

template<typename IN_T, typename IDX_T, cnnlScatterMode_t mode>
class MluDimScatterKernel final : public user_op::OpKernel {
 public:
  MluDimScatterKernel() = default;
  ~MluDimScatterKernel() override = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* src_tensor = ctx->Tensor4ArgNameAndIndex("src", 0);
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);

    const int32_t dim = ctx->Attr<int32_t>("dim");
    size_t out_bytes_size =
        out_tensor->shape_view().elem_cnt() * GetSizeOfDataType(out_tensor->data_type());

    const Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);

    if (input_tensor) {
      Memcpy<DeviceType::kMLU>(ctx->stream(), out_tensor->mut_dptr(), input_tensor->dptr(),
                               out_bytes_size);
    } else if (like_tensor) {
      Memset<DeviceType::kMLU>(ctx->stream(), out_tensor->mut_dptr(), 0, out_bytes_size);
    } else {
      UNIMPLEMENTED() << "Input tensor and like tensor cannot be empty simultaneously.";
    }

    CnnlTensorDescriptor src_desc(src_tensor), index_desc(index_tensor), out_desc(out_tensor);
    auto* cnnl_handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
    OF_CNNL_CHECK(cnnlScatter(cnnl_handle, dim, out_desc.desc(), out_tensor->dptr(),
                              index_desc.desc(), index_tensor->dptr(), src_desc.desc(),
                              src_tensor->dptr(), out_desc.desc(), out_tensor->mut_dptr(), mode));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_DIM_SCATTER_LIKE_KERNEL(dtype, itype)                               \
  REGISTER_USER_KERNEL("dim_scatter_add_like")                                           \
      .SetCreateFn<MluDimScatterKernel<dtype, itype, CNNL_SCATTER_ADD>>()                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                    \
                       && (user_op::HobDataType("like", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("index", 0) == GetDataType<itype>::value));

REGISTER_MLU_DIM_SCATTER_LIKE_KERNEL(float, int32_t)
REGISTER_MLU_DIM_SCATTER_LIKE_KERNEL(float16, int32_t)
REGISTER_MLU_DIM_SCATTER_LIKE_KERNEL(int32_t, int32_t)
REGISTER_MLU_DIM_SCATTER_LIKE_KERNEL(float, int64_t)
REGISTER_MLU_DIM_SCATTER_LIKE_KERNEL(float16, int64_t)
REGISTER_MLU_DIM_SCATTER_LIKE_KERNEL(int32_t, int64_t)

#undef REGISTER_MLU_DIM_SCATTER_LIKE_KERNEL

#define REGISTER_MLU_DIM_SCATTER_KERNEL_IMPL(op_type, dtype, itype, mode)                 \
  REGISTER_USER_KERNEL(#op_type)                                                          \
      .SetCreateFn<MluDimScatterKernel<dtype, itype, mode>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("index", 0) == GetDataType<itype>::value));

#define REGISTER_MLU_DIM_SCATTER_KERNEL(dtype, itype)                                   \
  REGISTER_MLU_DIM_SCATTER_KERNEL_IMPL(dim_scatter_add, dtype, itype, CNNL_SCATTER_ADD) \
  REGISTER_MLU_DIM_SCATTER_KERNEL_IMPL(dim_scatter_update, dtype, itype, CNNL_SCATTER)

REGISTER_MLU_DIM_SCATTER_KERNEL(float, int32_t)
REGISTER_MLU_DIM_SCATTER_KERNEL(float16, int32_t)
REGISTER_MLU_DIM_SCATTER_KERNEL(int32_t, int32_t)
REGISTER_MLU_DIM_SCATTER_KERNEL(float, int64_t)
REGISTER_MLU_DIM_SCATTER_KERNEL(float16, int64_t)
REGISTER_MLU_DIM_SCATTER_KERNEL(int32_t, int64_t)

#undef REGISTER_MLU_DIM_SCATTER_KERNEL
#undef REGISTER_MLU_DIM_SCATTER_KERNEL_IMPL

}  // namespace user_op
}  // namespace oneflow
