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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/primitive/include/copy_nd.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<primitive::CopyNd> NewCopyNdPrimitive(Context* ctx) {
  return primitive::NewPrimitive<primitive::CopyNdFactory>(ctx->device_type(), 2);
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
    const ShapeView in_shape = in->shape(); 
    auto primitive = NewCopyNdPrimitive(ctx);

    DimVector dst_shape = {in_shape.Count(0, dim), length, in_shape.Count(dim + 1)};
    DimVector dst_pos_vec = {0, 0, 0};

    DimVector src_shape = {in_shape.Count(0, dim), in_shape.At(dim), in_shape.Count(dim + 1)};
    DimVector src_pos_vec = {0, start, 0};
    DimVector extent_vec = {in_shape.Count(0, dim), length, in_shape.Count(dim + 1)};
    primitive->Launch(ctx->stream_ctx(), out->data_type(), 3, out->mut_dptr(),
                      dst_shape.data(), dst_pos_vec.data(), in->dptr(), src_shape.data(),
                      src_pos_vec.data(), extent_vec.data());
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
    
    auto primitive = NewCopyNdPrimitive(ctx);
    const ShapeView dx_shape = dx->shape(); 

    DimVector dst_shape = {dx_shape.Count(0, dim), dx_shape.At(dim), dx_shape.Count(dim + 1)};
    DimVector dst_pos_vec = {0, start, 0};

    DimVector src_shape = {dx_shape.Count(0, dim), length, dx_shape.Count(dim + 1)};
    DimVector src_pos_vec = {0, 0, 0};
    DimVector extent_vec = {dx_shape.Count(0, dim), length, dx_shape.Count(dim + 1)};

    primitive->Launch(ctx->stream_ctx(), dx->data_type(), 3, dx->mut_dptr(),
                      dst_shape.data(), dst_pos_vec.data(), dy->dptr(), src_shape.data(),
                      src_pos_vec.data(), extent_vec.data());
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
