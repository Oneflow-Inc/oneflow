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

namespace {

template<DeviceType device_type, typename T, typename K>
void GetReduceSumWorkspaceSizeInBytes(int64_t n, int64_t m, int64_t* workspace_size_in_bytes) {
  IndexedSlicesReduceSumKernelUtil<device_type, K, T, int64_t>::GetReduceSumWorkspaceSizeInBytes(
      nullptr, n, m, workspace_size_in_bytes);
}

struct SwitchUtil final {
#define SWITCH_ENTRY(func_name, device_type, T, K) func_name<device_type, T, K>
  DEFINE_STATIC_SWITCH_FUNC(void, GetReduceSumWorkspaceSizeInBytes, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));
#undef SWITCH_ENTRY
};

}  // namespace

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
    SwitchUtil::SwitchGetReduceSumWorkspaceSizeInBytes(
        SwitchCase(device_type, x_values->data_type(), x_indices->data_type()), n, m,
        &workspace_size_in_bytes);
    return workspace_size_in_bytes;
  };
}

#define REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(device_type, data_type, indices_type)  \
  REGISTER_USER_KERNEL("indexed_slices_reduce_sum")                                      \
      .SetCreateFn<IndexedSlicesReduceSumKernel<device_type, data_type, indices_type>>() \
      .SetIsMatchedHob(                                                                  \
          (user_op::HobDeviceTag() == device_type)                                       \
          & (user_op::HobDataType("x_values", 0) == GetDataType<data_type>::value)       \
          & (user_op::HobDataType("x_indices", 0) == GetDataType<indices_type>::value))  \
      .SetInferTmpSizeFn(GenInferTmpSizeFn<device_type, data_type, indices_type>());

REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(DeviceType::kGPU, float, int32_t)
REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(DeviceType::kGPU, float, int64_t)
REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(DeviceType::kGPU, double, int32_t)
REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(DeviceType::kGPU, double, int64_t)
REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(DeviceType::kCPU, float, int32_t)
REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(DeviceType::kCPU, float, int64_t)
REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(DeviceType::kCPU, double, int32_t)
REGISTER_INDEXED_SLICES_REDUCE_SUM_KERNEL(DeviceType::kCPU, double, int64_t)

}  // namespace oneflow
