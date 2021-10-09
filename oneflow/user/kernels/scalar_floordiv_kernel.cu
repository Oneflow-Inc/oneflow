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
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace user_op {

template<typename T>
struct ScalarFloorDivFunctor {
  OF_DEVICE_FUNC explicit ScalarFloorDivFunctor(T scalar) : scalar(scalar) {}
  __device__ T operator()(T x) const { return floor(fdividef(x, scalar)); }
  const T scalar;
};

template<>
struct ScalarFloorDivFunctor<int8_t> {
  OF_DEVICE_FUNC explicit ScalarFloorDivFunctor(int8_t scalar) : scalar(scalar) {}
  __device__ int8_t operator()(int8_t x) const { return x / scalar; }
  const int8_t scalar;
};

template<>
struct ScalarFloorDivFunctor<uint8_t> {
  OF_DEVICE_FUNC explicit ScalarFloorDivFunctor(uint8_t scalar) : scalar(scalar) {}
  __device__ uint8_t operator()(uint8_t x) const { return x / scalar; }
  const uint8_t scalar;
};

template<>
struct ScalarFloorDivFunctor<int32_t> {
  OF_DEVICE_FUNC explicit ScalarFloorDivFunctor(int32_t scalar) : scalar(scalar) {}
  __device__ int32_t operator()(int32_t x) const { return x / scalar; }
  const int32_t scalar;
};

template<>
struct ScalarFloorDivFunctor<int64_t> {
  OF_DEVICE_FUNC explicit ScalarFloorDivFunctor(int64_t scalar) : scalar(scalar) {}
  __device__ int64_t operator()(int64_t x) const { return x / scalar; }
  const int64_t scalar;
};

template<DeviceType device_type, typename T>
class GpuScalarFloorDivKernel final : public OpKernel {
 public:
  GpuScalarFloorDivKernel() = default;
  ~GpuScalarFloorDivKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }

    const int64_t elem_cnt = in_tensor->shape().elem_cnt();
    OF_CUDA_CHECK(
        (oneflow::cuda::elementwise::Unary(ScalarFloorDivFunctor<T>(scalar_operand), elem_cnt,
                                           out_ptr, in_ptr, ctx->device_ctx()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_SCALAR_FLOOR_DIV_KERNEL(device, dtype)  \
  REGISTER_USER_KERNEL("scalar_floordiv")                    \
      .SetCreateFn<GpuScalarFloorDivKernel<device, dtype>>() \
      .SetIsMatchedHob((HobDeviceTag() == device)            \
                       & (HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_SCALAR_FLOOR_DIV_KERNEL(DeviceType::kGPU, uint8_t);
REGISTER_GPU_SCALAR_FLOOR_DIV_KERNEL(DeviceType::kGPU, int8_t);
REGISTER_GPU_SCALAR_FLOOR_DIV_KERNEL(DeviceType::kGPU, int32_t);
REGISTER_GPU_SCALAR_FLOOR_DIV_KERNEL(DeviceType::kGPU, int64_t);
REGISTER_GPU_SCALAR_FLOOR_DIV_KERNEL(DeviceType::kGPU, float);
REGISTER_GPU_SCALAR_FLOOR_DIV_KERNEL(DeviceType::kGPU, double);

}  // namespace user_op

}  // namespace oneflow
