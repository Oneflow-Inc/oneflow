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
#include "oneflow/user/kernels/narrow_util.h"

namespace oneflow {

namespace user_op {

namespace {

Shape GetFlatShape(const ShapeView& shape, int64_t dim) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(dim, 0);
  CHECK_LT(dim, shape.NumAxes());
  return Shape({shape.Count(0, dim), shape.At(dim), shape.Count(dim + 1)});
}

}  // namespace

template<DeviceType device_type, typename T>
class NarrowKernel final : public user_op::OpKernel {
 public:
  NarrowKernel() = default;
  ~NarrowKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t& dim = ctx->Attr<int64_t>("dim");
    const int64_t& start = ctx->Attr<int64_t>("start");
    const int64_t& length = ctx->Attr<int64_t>("length");
    NarrowKernelUtil<device_type, T>::Forward(ctx->device_ctx(), in->dptr<T>(),
                                              GetFlatShape(in->shape(), dim), out->mut_dptr<T>(),
                                              start, length);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class NarrowGradKernel final : public user_op::OpKernel {
 public:
  NarrowGradKernel() = default;
  ~NarrowGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t& dim = ctx->Attr<int64_t>("dim");
    const int64_t& start = ctx->Attr<int64_t>("start");
    const int64_t& length = ctx->Attr<int64_t>("length");
    size_t dx_byte_size = dx->shape().elem_cnt() * sizeof(T);
    Memset<device_type>(ctx->device_ctx(), dx->mut_dptr<T>(), 0, dx_byte_size);
    NarrowKernelUtil<device_type, T>::Backward(ctx->device_ctx(), dy->dptr<T>(),
                                               GetFlatShape(dx->shape(), dim), dx->mut_dptr<T>(),
                                               start, length);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_NARROW_KERNELS(device, dtype)                                               \
  REGISTER_USER_KERNEL("narrow").SetCreateFn<NarrowKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                    \
      & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));                       \
  REGISTER_USER_KERNEL("narrow_grad")                                                        \
      .SetCreateFn<NarrowGradKernel<device, dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                   \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#define REGISTER_NARROW_KERNELS_WITH_DEVICE(device) \
  REGISTER_NARROW_KERNELS(device, float)            \
  REGISTER_NARROW_KERNELS(device, double)           \
  REGISTER_NARROW_KERNELS(device, int32_t)          \
  REGISTER_NARROW_KERNELS(device, int64_t)          \
  REGISTER_NARROW_KERNELS(device, int8_t)           \
  REGISTER_NARROW_KERNELS(device, uint8_t)

REGISTER_NARROW_KERNELS_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_NARROW_KERNELS_WITH_DEVICE(DeviceType::kGPU)
#endif

}  // namespace user_op

}  // namespace oneflow
