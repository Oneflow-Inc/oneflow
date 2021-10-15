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
#include "oneflow/core/primitive/include/permute.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace user_op {

template<typename Context>
std::unique_ptr<primitive::Permute> NewPermutePrimitive(Context* ctx) {
  const int32_t num_dims = ctx->TensorDesc4ArgNameAndIndex("output", 0)->shape().NumAxes();
  return primitive::NewPrimitive<primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

class TransposeKernel final : public OpKernel, public user_op::CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransposeKernel);
  TransposeKernel() = default;
  ~TransposeKernel() override = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    auto primitive = NewPermutePrimitive(ctx);
    CHECK(primitive);

    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("input", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto& perm = ctx->Attr<std::vector<int32_t>>("perm");
    const ShapeView& in_shape = tensor_in->shape();
    DataType dtype = tensor_out->data_type();
    size_t num_dims = tensor_in->shape().NumAxes();
    DimVector src_dims;
    in_shape.ToDimVector(&src_dims);

    primitive->Launch(ctx->stream_ctx(), dtype, num_dims, src_dims.data(), tensor_in->dptr(),
                      perm.data(), tensor_out->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TRANSPOSE_KERNEL(device, dtype)                                         \
  REGISTER_USER_KERNEL("transpose")                                                      \
      .SetCreateFn<TransposeKernel>()                                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("output", 0) == GetDataType<dtype>::value));

REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, uint8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, float)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, uint8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, float)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, double)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, float16)
#endif

}  // namespace user_op
}  // namespace oneflow
