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
#include "oneflow/user/kernels/unique_kernel_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {
template<DeviceType device_type, typename T, typename U>
void GetUniqueWorkspaceSizeInBytes(int64_t n, int64_t* workspace_size_in_bytes) {
  UniqueKernelUtil<device_type, T, U>::GetUniqueWorkspaceSizeInBytes(nullptr, n,
                                                                     workspace_size_in_bytes);
}

template<DeviceType device_type, typename T, typename U>
void GetUniqueWithCountsWorkspaceSizeInBytes(int64_t n, int64_t* workspace_size_in_bytes) {
  UniqueKernelUtil<device_type, T, U>::GetUniqueWithCountsWorkspaceSizeInBytes(
      nullptr, n, workspace_size_in_bytes);
}

struct SwitchUtil final {
#define SWITCH_ENTRY(func_name, device_type, T, U) func_name<device_type, T, U>
  DEFINE_STATIC_SWITCH_FUNC(void, GetUniqueWorkspaceSizeInBytes, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));
  DEFINE_STATIC_SWITCH_FUNC(void, GetUniqueWithCountsWorkspaceSizeInBytes, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));
#undef SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T, typename K>
class UniqueWithCountsKernel final : public user_op::OpKernel {
 public:
  UniqueWithCountsKernel() = default;
  ~UniqueWithCountsKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* idx = ctx->Tensor4ArgNameAndIndex("idx", 0);
    user_op::Tensor* count = ctx->Tensor4ArgNameAndIndex("count", 0);
    user_op::Tensor* num_unique = ctx->Tensor4ArgNameAndIndex("num_unique", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    void* tmp_ptr = tmp ? tmp->mut_dptr() : nullptr;
    int64_t tmp_size = tmp ? tmp->shape().elem_cnt() * GetSizeOfDataType(tmp->data_type()) : 0;
    UniqueKernelUtil<device_type, T, K>::UniqueWithCounts(
        ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(), num_unique->mut_dptr<K>(),
        y->mut_dptr<T>(), idx->mut_dptr<K>(), count->mut_dptr<K>(), tmp_ptr, tmp_size);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
template<DeviceType device_type, typename T, typename K>
user_op::InferTmpSizeFn GenInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const auto* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
    auto out_idx = ctx->Attr<DataType>("out_idx");
    int64_t workspace_size_in_bytes;
    SwitchUtil::SwitchGetUniqueWithCountsWorkspaceSizeInBytes(
        SwitchCase(device_type, x->data_type(), out_idx), x->shape().elem_cnt(),
        &workspace_size_in_bytes);
    return workspace_size_in_bytes;
  };
}

#define REGISTER_UNIQUE_WITH_COUNTS_KERNEL(device_type, data_type, indices_type)               \
  REGISTER_USER_KERNEL("unique_with_counts")                                                   \
      .SetCreateFn<UniqueWithCountsKernel<device_type, data_type, indices_type>>()             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type)                                \
                       & (user_op::HobDataType("x", 0) == GetDataType<data_type>::value)       \
                       & (user_op::HobDataType("idx", 0) == GetDataType<indices_type>::value)) \
      .SetInferTmpSizeFn(GenInferTmpSizeFn<device_type, data_type, indices_type>());

REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, float, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, float, int64_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, double, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, double, int64_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, int8_t, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, int8_t, int64_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, int32_t, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, int32_t, int64_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, int64_t, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kGPU, int64_t, int64_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, float, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, float, int64_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, double, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, double, int64_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, int8_t, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, int8_t, int64_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, int32_t, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, int32_t, int64_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, int64_t, int32_t)
REGISTER_UNIQUE_WITH_COUNTS_KERNEL(DeviceType::kCPU, int64_t, int64_t)

}  // namespace oneflow
