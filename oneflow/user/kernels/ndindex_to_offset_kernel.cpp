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
#include "oneflow/user/kernels/ndindex_to_offset_util.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class NdIndexToOffsetKernel final : public OpKernel {
 public:
  NdIndexToOffsetKernel() = default;
  ~NdIndexToOffsetKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* dims_tensor = ctx->Tensor4ArgNameAndIndex("dims", 0);
    const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);

    int ndim = dims_tensor->shape().elem_cnt();         // dims_tensor [8, 8, 8] -> ndim = 3
    int32_t in_num = index_tensor->shape().elem_cnt();  // index_tensor [4, 4, 2] -> in_num = 3
    CHECK_EQ(ndim, in_num);                             // Check the numbers of shape is equal

    // // To avoid the dims of index is larger than the oneflow default max ndim.
    CHECK_LE(in_num, oneflow::index_max_ndims);

    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();

    NdIndexToOffsetFunctor<device_type, T>()(
        ctx->device_ctx(), in_num, ndim, index_tensor->dptr<T>(), dims_tensor->dptr<T>(), output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_NDINDEX_TO_OFFSET_KERNEL(device, dtype)   \
  REGISTER_USER_KERNEL("ndindex_to_offset")                \
      .SetCreateFn<NdIndexToOffsetKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("dims", 0) == GetDataType<dtype>::value));

#define REGISTER_NDINDEX_TO_OFFSET_KERNELS_WITH_DEVICE(device) \
  REGISTER_NDINDEX_TO_OFFSET_KERNEL(device, int32_t)           \
  REGISTER_NDINDEX_TO_OFFSET_KERNEL(device, int64_t)

// Register CPU version
REGISTER_NDINDEX_TO_OFFSET_KERNELS_WITH_DEVICE(DeviceType::kCPU);

// Register GPU version
#ifdef WITH_CUDA
REGISTER_NDINDEX_TO_OFFSET_KERNELS_WITH_DEVICE(DeviceType::kGPU);
#endif

}  // namespace user_op
}  // namespace oneflow
