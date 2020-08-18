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
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class TransposeKernel final : public OpKernel {
 public:
  TransposeKernel() = default;
  ~TransposeKernel() override = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("input", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto& perm = ctx->Attr<std::vector<int32_t>>("perm");
    using PackType = int64_t;
    const size_t num_elem_per_pack = sizeof(PackType) / sizeof(T);
    const ShapeView& in_shape = tensor_in->shape();
    const ShapeView& out_shape = tensor_out->shape();
    if (num_elem_per_pack != 1 && perm.back() == perm.size() - 1
        && in_shape.At(in_shape.NumAxes() - 1) % num_elem_per_pack == 0) {
      CHECK_EQ(in_shape.At(in_shape.NumAxes() - 1), out_shape.At(out_shape.NumAxes() - 1));
      DimVector packed_in_dim_vec;
      in_shape.ToDimVector(&packed_in_dim_vec);
      packed_in_dim_vec.back() /= num_elem_per_pack;
      const Shape packed_in_shape(packed_in_dim_vec);
      DimVector packed_out_dim_vec;
      out_shape.ToDimVector(&packed_out_dim_vec);
      packed_out_dim_vec.back() /= num_elem_per_pack;
      const Shape packed_out_shape(packed_out_dim_vec);
      NewKernelUtil<device_type>::Transpose(
          ctx->device_ctx(), packed_in_shape.NumAxes(), packed_in_shape, packed_out_shape, perm,
          packed_in_shape.elem_cnt(), reinterpret_cast<const PackType*>(tensor_in->dptr<T>()),
          reinterpret_cast<PackType*>(tensor_out->mut_dptr<T>()));
    } else {
      NewKernelUtil<device_type>::Transpose(ctx->device_ctx(), in_shape.NumAxes(), in_shape,
                                            tensor_out->shape(), perm, in_shape.elem_cnt(),
                                            tensor_in->dptr<T>(), tensor_out->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TRANSPOSE_KERNEL(device, dtype)                                         \
  REGISTER_USER_KERNEL("transpose")                                                      \
      .SetCreateFn<TransposeKernel<device, dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("output", 0) == GetDataType<dtype>::value));

REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, float)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, float)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, double)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, float16)
#endif

}  // namespace user_op
}  // namespace oneflow
