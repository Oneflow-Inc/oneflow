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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/include/primitive/broadcast_matmul.h"

namespace oneflow {

namespace {

using namespace ep::primitive;

template<DeviceType device_type, typename T>
class DotKernel final : public user_op::OpKernel {
 public:
  DotKernel() = default;
  ~DotKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = x->shape().elem_cnt();
    CHECK(n <= INT_MAX);

    const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
    auto primitive = NewPrimitive<BroadcastMatmulFactory>(
        ctx->device_type(), data_type, BlasTransposeType::N, BlasTransposeType::N, 2);

    std::vector<int64_t> dim1{1, n};
    std::vector<int64_t> dim2{n, 1};
    std::vector<int64_t> dim3{1, 1};
    primitive->Launch(ctx->stream(), 1, 2, dim1.data(), x, 2, dim2.data(), y, 1, 2, dim3.data(),
                      out);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DOT_KERNEL(device, dtype)                                             \
  REGISTER_USER_KERNEL("dot").SetCreateFn<DotKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                             \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_DOT_KERNEL(DeviceType::kCPU, float)
REGISTER_DOT_KERNEL(DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_DOT_KERNEL(DeviceType::kGPU, float)
REGISTER_DOT_KERNEL(DeviceType::kGPU, double)
#endif

}  // namespace

}  // namespace oneflow
