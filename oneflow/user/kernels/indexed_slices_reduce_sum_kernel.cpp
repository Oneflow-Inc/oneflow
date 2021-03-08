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
#include "oneflow/user/kernels/indexed_slices_reduce_sum_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class IndexedSlicesReduceSumKernel final : public user_op::OpKernel {
 public:
  IndexedSlicesReduceSumKernel() = default;
  ~IndexedSlicesReduceSumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_indices = ctx->Tensor4ArgNameAndIndex("x_indices", 0);
    const user_op::Tensor* x_values = ctx->Tensor4ArgNameAndIndex("x_values", 0);
    user_op::Tensor* y_indices = ctx->Tensor4ArgNameAndIndex("y_indices", 0);
    user_op::Tensor* y_values = ctx->Tensor4ArgNameAndIndex("y_values", 0);
    user_op::Tensor* num_unique = ctx->Tensor4ArgNameAndIndex("num_unique", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    void* tmp_ptr = tmp ? tmp->mut_dptr() : nullptr;
    int64_t tmp_size = tmp ? tmp->shape().elem_cnt() * GetSizeOfDataType(tmp->data_type()) : 0;
    const int64_t n = x_indices->shape().elem_cnt();
    const int64_t m = x_values->shape().elem_cnt() / n;
    IndexedSlicesReduceSumKernelUtil<device_type, K, T, int64_t>::ReduceSum(
        ctx->device_ctx(), n, m, x_indices->dptr<K>(), x_values->dptr<T>(),
        num_unique->mut_dptr<int64_t>(), y_indices->mut_dptr<K>(), y_values->mut_dptr<T>(), tmp_ptr,
        tmp_size);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename K>
user_op::InferTmpSizeFn GenInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const auto* x_indices = ctx->TensorDesc4ArgNameAndIndex("x_indices", 0);
    const auto* x_values = ctx->TensorDesc4ArgNameAndIndex("x_values", 0);
    const int64_t n = x_indices->shape().elem_cnt();
    const int64_t m = x_values->shape().elem_cnt() / n;
    int64_t workspace_size_in_bytes;
    IndexedSlicesReduceSumKernelUtil<device_type, K, T, int64_t>::GetReduceSumWorkspaceSizeInBytes(
        nullptr, n, m, &workspace_size_in_bytes);
    return workspace_size_in_bytes;
  };
}

#define REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(device_type_v, data_type_pair,                 \
                                                  indices_type_pair)                             \
  REGISTER_USER_KERNEL("indexed_slices_reduce_sum")                                              \
      .SetCreateFn<IndexedSlicesReduceSumKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair), \
                                                OF_PP_PAIR_FIRST(indices_type_pair)>>()          \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceTag() == ToString(device_type_v))                                   \
          & (user_op::HobDataType("x_values", 0) == OF_PP_PAIR_SECOND(data_type_pair))           \
          & (user_op::HobDataType("x_indices", 0) == OF_PP_PAIR_SECOND(indices_type_pair)))      \
      .SetInferTmpSizeFn(GenInferTmpSizeFn<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),      \
                                           OF_PP_PAIR_FIRST(indices_type_pair)>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
