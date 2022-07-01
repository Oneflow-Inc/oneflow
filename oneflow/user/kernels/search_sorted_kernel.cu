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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/user/kernels/search_sorted_kernel_util.h"

namespace oneflow {

template<typename T, typename K>
__global__ void DoSearchSortedLogical(int32_t instance_num, bool is_sequence_1d,
                                      K values_shape_last, K sequence_shape_last, bool right,
                                      const T* values_ptr, const T* sequence_ptr, K* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    K start_bd = is_sequence_1d ? 0 : i / values_shape_last * sequence_shape_last;
    K end_bd = start_bd + sequence_shape_last;
    K pos = !right
                ? cus_lower_bound<T, K>(start_bd, end_bd, values_ptr[i], sequence_ptr) - start_bd
                : cus_upper_bound<T, K>(start_bd, end_bd, values_ptr[i], sequence_ptr) - start_bd;
    out_ptr[i] = pos;
  }
}

template<typename T, typename K>
__global__ void DoSearchSortedScalarLogical(K sequence_shape_last, bool right, const T values,
                                            const T* sequence_ptr, K* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, 1) {
    K pos = !right ? cus_lower_bound<T, K>(0, sequence_shape_last, values, sequence_ptr)
                   : cus_upper_bound<T, K>(0, sequence_shape_last, values, sequence_ptr);
    out_ptr[0] = pos;
  }
}

template<typename T, typename K>
class GpuSearchSortedKernel final : public user_op::OpKernel {
 public:
  GpuSearchSortedKernel() = default;
  ~GpuSearchSortedKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* sorted_sequence = ctx->Tensor4ArgNameAndIndex("sorted_sequence", 0);
    const user_op::Tensor* values = ctx->Tensor4ArgNameAndIndex("values", 0);

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const bool& right = ctx->Attr<bool>("right");
    const T* values_ptr = values->dptr<T>();
    const T* sequence_ptr = sorted_sequence->dptr<T>();
    K* out_ptr = out->mut_dptr<K>();
    const int32_t instance_num = values->shape_view().elem_cnt();
    bool is_values_scalar = values->shape_view().NumAxes() == 0;
    bool is_sequence_1d = (sorted_sequence->shape_view().NumAxes() == 1);
    K values_shape_last =
        is_values_scalar ? 1 : values->shape_view().At(values->shape_view().NumAxes() - 1);
    K sequence_shape_last =
        sorted_sequence->shape_view().At(sorted_sequence->shape_view().NumAxes() - 1);
    RUN_CUDA_KERNEL((DoSearchSortedLogical<T, K>), ctx->stream(), instance_num, instance_num,
                    is_sequence_1d, values_shape_last, sequence_shape_last, right, values_ptr,
                    sequence_ptr, out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_SEARCH_SORTED_KERNEL(in_dtype, out_dtype)                              \
  REGISTER_USER_KERNEL("searchsorted")                                                      \
      .SetCreateFn<                                                                         \
          GpuSearchSortedKernel<OF_PP_PAIR_FIRST(in_dtype), OF_PP_PAIR_FIRST(out_dtype)>>() \
      .SetIsMatchedHob(                                                                     \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                   \
          && (user_op::HobDataType("sorted_sequence", 0) == OF_PP_PAIR_SECOND(in_dtype))    \
          && (user_op::HobDataType("values", 0) == OF_PP_PAIR_SECOND(in_dtype))             \
          && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(out_dtype)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GPU_SEARCH_SORTED_KERNEL, ARITHMETIC_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

template<typename T, typename K>
class GpuSearchSortedScalarKernel final : public user_op::OpKernel {
 public:
  GpuSearchSortedScalarKernel() = default;
  ~GpuSearchSortedScalarKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* sorted_sequence = ctx->Tensor4ArgNameAndIndex("sorted_sequence", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const bool& right = ctx->Attr<bool>("right");
    const T& values = static_cast<T>(ctx->Attr<double>("values"));

    const T* sequence_ptr = sorted_sequence->dptr<T>();
    K* out_ptr = out->mut_dptr<K>();
    K sequence_shape_last = sorted_sequence->shape_view().At(0);
    RUN_CUDA_KERNEL((DoSearchSortedScalarLogical<T, K>), ctx->stream(), 1, sequence_shape_last,
                    right, values, sequence_ptr, out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_SEARCH_SORTED_SCALAR_KERNEL(in_dtype, out_dtype)                             \
  REGISTER_USER_KERNEL("searchsorted_scalar")                                                     \
      .SetCreateFn<                                                                               \
          GpuSearchSortedScalarKernel<OF_PP_PAIR_FIRST(in_dtype), OF_PP_PAIR_FIRST(out_dtype)>>() \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("sorted_sequence", 0) == OF_PP_PAIR_SECOND(in_dtype))          \
          && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(out_dtype)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GPU_SEARCH_SORTED_SCALAR_KERNEL, ARITHMETIC_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
