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
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace {

std::unique_ptr<ep::primitive::Cast> NewCastPrimitive(DataType from, DataType to) {
  return ep::primitive::NewPrimitive<ep::primitive::CastFactory>(DeviceType::kMLU, from, to);
}

template<typename T>
DataType GetReduceComputeDataType() {
  return GetDataType<T>::value;
}

#define GET_REDUCE_COMPUTE_DATA_TYPE(T, D) \
  template<>                               \
  DataType GetReduceComputeDataType<T>() { \
    return GetDataType<D>::value;          \
  }

GET_REDUCE_COMPUTE_DATA_TYPE(uint32_t, int32_t)
GET_REDUCE_COMPUTE_DATA_TYPE(int64_t, int32_t)
GET_REDUCE_COMPUTE_DATA_TYPE(uint64_t, int32_t)

#undef GET_REDUCE_COMPUTE_DATA_TYPE

template<cnnlReduceOp_t mode, typename T>
class MulReduceKernel final : public user_op::OpKernel {
 public:
  MulReduceKernel() = default;
  ~MulReduceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");

    CnnlReduceDescriptor reduce_desc;
    CnnlTensorDescriptor input_desc, output_desc;
    CnnlWorkspace cast_workspace(ctx->stream()->As<ep::MluStream>());
    DataType compute_dtype = GetReduceComputeDataType<T>();

    const void* input_ptr = input->dptr();
    void* output_ptr = output->mut_dptr();

    if (compute_dtype != GetDataType<T>::value) {
      int element_size = GetSizeOfDataType(compute_dtype);
      size_t input_count = input->shape_view().elem_cnt() * element_size;
      size_t output_count = output->shape_view().elem_cnt() * element_size;
      cast_workspace.resize(input_count + output_count);
      input_ptr = cast_workspace.dptr();
      output_ptr = static_cast<char*>(cast_workspace.dptr()) + input_count;

      auto cast_input = NewCastPrimitive(GetDataType<T>::value, compute_dtype);
      cast_input->Launch(ctx->stream(), input->dptr(), cast_workspace.dptr(),
                         input_count / element_size);
      cast_input->Launch(ctx->stream(), output->dptr(), output_ptr, output_count / element_size);
    }

    auto cnnl_dtype = ConvertToCnnlDataType(compute_dtype);
    input_desc.set(input->shape_view().NumAxes(), input->shape_view().data(), cnnl_dtype);

    auto reduce_indices = CNNL_REDUCE_NO_INDICES;
    auto reduce_indices_type = CNNL_32BIT_INDICES;

    if (axis.size() == input->shape_view().NumAxes()) {
      std::vector<int32_t> full_reduce(1, -1);
      std::vector<int32_t> fake_size(input->shape_view().NumAxes(), 1);
      reduce_desc.set(cnnl_dtype, full_reduce, mode, reduce_indices, reduce_indices_type);
      output_desc.set(fake_size.size(), fake_size.data(), cnnl_dtype);
    } else {
      reduce_desc.set(cnnl_dtype, axis, mode, reduce_indices, reduce_indices_type);
      output_desc.set(output->shape_view().NumAxes(), output->shape_view().data(), cnnl_dtype);
    }
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                               input_desc.desc(), output_desc.desc(),
                                               reduce_desc.mut_desc(), &workspace_size));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);

    OF_CNNL_CHECK(cnnlReduce(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), reduce_desc.desc(),
                             workspace.dptr(), workspace_size, nullptr, input_desc.desc(),
                             input_ptr, 0, nullptr, nullptr, output_desc.desc(), output_ptr));

    if (compute_dtype != GetDataType<T>::value) {
      auto cast_output = NewCastPrimitive(compute_dtype, GetDataType<T>::value);
      cast_output->Launch(ctx->stream(), output_ptr, output->mut_dptr(),
                          output->shape_view().elem_cnt());
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_REDUCE_SUM_KERNEL(dtype)                 \
  REGISTER_USER_KERNEL("reduce_sum")                          \
      .SetCreateFn<MulReduceKernel<CNNL_REDUCE_ADD, dtype>>() \
      .SetIsMatchedHob(                                       \
          (user_op::HobDeviceType() == DeviceType::kMLU)      \
          && (user_op::HobDataType("output_tensor", 0) == GetDataType<dtype>::value));

#define REGISTER_MLU_REDUCE_MAX_KERNEL(dtype)                 \
  REGISTER_USER_KERNEL("reduce_max")                          \
      .SetCreateFn<MulReduceKernel<CNNL_REDUCE_MAX, dtype>>() \
      .SetIsMatchedHob(                                       \
          (user_op::HobDeviceType() == DeviceType::kMLU)      \
          && (user_op::HobDataType("output_tensor", 0) == GetDataType<dtype>::value));

#define REGISTER_MLU_REDUCE_MIN_KERNEL(dtype)                 \
  REGISTER_USER_KERNEL("reduce_min")                          \
      .SetCreateFn<MulReduceKernel<CNNL_REDUCE_MIN, dtype>>() \
      .SetIsMatchedHob(                                       \
          (user_op::HobDeviceType() == DeviceType::kMLU)      \
          && (user_op::HobDataType("output_tensor", 0) == GetDataType<dtype>::value));

#define REGISTER_MLU_REDUCE_KERNEL(T) \
  REGISTER_MLU_REDUCE_SUM_KERNEL(T)   \
  REGISTER_MLU_REDUCE_MAX_KERNEL(T)   \
  REGISTER_MLU_REDUCE_MIN_KERNEL(T)

REGISTER_MLU_REDUCE_KERNEL(float)
REGISTER_MLU_REDUCE_KERNEL(float16)
REGISTER_MLU_REDUCE_KERNEL(int32_t)
REGISTER_MLU_REDUCE_KERNEL(uint32_t)
REGISTER_MLU_REDUCE_KERNEL(int64_t)
REGISTER_MLU_REDUCE_KERNEL(uint64_t)

}  // namespace
}  // namespace oneflow
