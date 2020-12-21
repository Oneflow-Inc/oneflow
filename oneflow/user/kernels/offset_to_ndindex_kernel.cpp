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
#include "oneflow/core/common/data_type.h"
#include "oneflow/user/kernels/offset_to_ndindex_util.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class OffsetToNdIndexKernel final : public OpKernel {
 public:
  OffsetToNdIndexKernel() = default;
  ~OffsetToNdIndexKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* offset_tensor = ctx->Tensor4ArgNameAndIndex("offset", 0);
    const Tensor* dims_tensor = ctx->Tensor4ArgNameAndIndex("dims", 0);

    int32_t dims_num = dims_tensor->shape().elem_cnt();

    // To avoid the dims of `dims` is larger than the oneflow default max ndim.
    CHECK_LE(dims_num, oneflow::dim_max_ndims);

    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();

    OffsetToNdIndexFunctor<device_type, T>()(ctx->device_ctx(), dims_num, offset_tensor->dptr<T>(),
                                             dims_tensor->dptr<T>(), output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_OFFSET_TO_NDINDEX_KERNEL(device, dtype)   \
  REGISTER_USER_KERNEL("offset_to_ndindex")                \
      .SetCreateFn<OffsetToNdIndexKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("dims", 0) == GetDataType<dtype>::value));

#define REGISTER_OFFSET_TO_NDINDEX_KERNELS_WITH_DEVICE(device) \
  REGISTER_OFFSET_TO_NDINDEX_KERNEL(device, int32_t)           \
  REGISTER_OFFSET_TO_NDINDEX_KERNEL(device, int64_t)

// Register CPU version
REGISTER_OFFSET_TO_NDINDEX_KERNELS_WITH_DEVICE(DeviceType::kCPU);

// Register GPU version
#ifdef WITH_CUDA
REGISTER_OFFSET_TO_NDINDEX_KERNELS_WITH_DEVICE(DeviceType::kGPU);
#endif

}  // namespace user_op
}  // namespace oneflow
