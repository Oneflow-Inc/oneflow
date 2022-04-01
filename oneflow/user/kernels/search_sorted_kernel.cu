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

namespace oneflow {

template<typename input_t, typename output_t>
OF_DEVICE_FUNC output_t cus_lower_bound(output_t start, output_t end, const input_t val,
                                        const input_t* bd, const int64_t* sort) {
  const output_t orig_start = start;
  while (start < end) {
    const output_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
OF_DEVICE_FUNC output_t cus_upper_bound(output_t start, output_t end, const input_t val,
                                        const input_t* bd, const int64_t* sort) {
  const output_t orig_start = start;
  while (start < end) {
    const output_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
__global__ void DoSearchSortedLogical(int32_t instance_num, bool is_sequence_1d,
                                      output_t values_shape_last, output_t sequence_shape_last,
                                      bool right, const input_t* values_ptr,
                                      const input_t* sequence_ptr, const int64_t* sorter_ptr,
                                      output_t* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    output_t start_bd = is_sequence_1d ? 0 : i / values_shape_last * sequence_shape_last;
    output_t end_bd = start_bd + sequence_shape_last;
    output_t pos = !right ? cus_lower_bound<input_t, output_t>(start_bd, end_bd, values_ptr[i],
                                                               sequence_ptr, sorter_ptr)
                                - start_bd
                          : cus_upper_bound<input_t, output_t>(start_bd, end_bd, values_ptr[i],
                                                               sequence_ptr, sorter_ptr)
                                - start_bd;
    out_ptr[i] = pos;
  }
}

template<typename input_t, typename output_t>
__global__ void DoSearchSortedScalarLogical(output_t sequence_shape_last, bool right,
                                            const input_t values, const input_t* sequence_ptr,
                                            const int64_t* sorter_ptr, output_t* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, 1) {
    output_t pos = !right ? cus_lower_bound<input_t, output_t>(0, sequence_shape_last, values,
                                                               sequence_ptr, sorter_ptr)
                          : cus_upper_bound<input_t, output_t>(0, sequence_shape_last, values,
                                                               sequence_ptr, sorter_ptr);
    out_ptr[0] = pos;
  }
}

template<typename input_t, typename output_t>
class GpuSearchSortedKernel final : public user_op::OpKernel {
 public:
  GpuSearchSortedKernel() = default;
  ~GpuSearchSortedKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* sorted_sequence = ctx->Tensor4ArgNameAndIndex("sorted_sequence", 0);
    const user_op::Tensor* values = ctx->Tensor4ArgNameAndIndex("values", 0);
    const user_op::Tensor* sorter = ctx->Tensor4ArgNameAndIndex("sorter", 0);

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const bool& right = ctx->Attr<bool>("right");
    const input_t* values_ptr = values->dptr<input_t>();
    const input_t* sequence_ptr = sorted_sequence->dptr<input_t>();
    const int64_t* sorter_ptr = nullptr;
    if (sorter) { sorter_ptr = sorter->dptr<int64_t>(); }
    output_t* out_ptr = out->mut_dptr<output_t>();
    const int32_t instance_num = values->shape().elem_cnt();
    bool is_values_scalar = (values->shape().elem_cnt() == 1 && values->shape().NumAxes() == 0);
    bool is_sequence_1d = (sorted_sequence->shape().NumAxes() == 1);
    output_t values_shape_last =
        is_values_scalar ? 1 : values->shape().At(values->shape().NumAxes() - 1);
    output_t sequence_shape_last =
        sorted_sequence->shape().At(sorted_sequence->shape().NumAxes() - 1);
    RUN_CUDA_KERNEL((DoSearchSortedLogical<input_t, output_t>), ctx->stream(), instance_num,
                    instance_num, is_sequence_1d, values_shape_last, sequence_shape_last, right,
                    values_ptr, sequence_ptr, sorter_ptr, out_ptr);
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

template<typename input_t, typename output_t>
class GpuSearchSortedScalarKernel final : public user_op::OpKernel {
 public:
  GpuSearchSortedScalarKernel() = default;
  ~GpuSearchSortedScalarKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* sorted_sequence = ctx->Tensor4ArgNameAndIndex("sorted_sequence", 0);
    const user_op::Tensor* sorter = ctx->Tensor4ArgNameAndIndex("sorter", 0);
    const int64_t* sorter_ptr = nullptr;
    if (sorter) { sorter_ptr = sorter->dptr<int64_t>(); }
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const bool& right = ctx->Attr<bool>("right");
    const input_t& values = static_cast<input_t>(ctx->Attr<double>("values"));

    const input_t* sequence_ptr = sorted_sequence->dptr<input_t>();
    output_t* out_ptr = out->mut_dptr<output_t>();
    output_t sequence_shape_last = sorted_sequence->shape().At(0);
    RUN_CUDA_KERNEL((DoSearchSortedScalarLogical<input_t, output_t>), ctx->stream(), 1,
                    sequence_shape_last, right, values, sequence_ptr, sorter_ptr, out_ptr);
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
