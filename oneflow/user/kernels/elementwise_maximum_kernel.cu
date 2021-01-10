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
#ifdef WITH_CUDA
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {
namespace user_op {

template<typename T>
struct MaximumFunctor {
  OF_DEVICE_FUNC T operator()(T x, T y) const { return x > y ? x : y; }
};

template<typename T>
class GpuElementwiseMaximumKernel final : public user_op::OpKernel {
 public:
  GpuElementwiseMaximumKernel() = default;
  ~GpuElementwiseMaximumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    int64_t n = tensor_x->shape().elem_cnt();

    OF_CUDA_CHECK(cuda::elementwise::Binary(MaximumFunctor<T>(), n, tensor_z->mut_dptr<T>(),
                                            tensor_x->dptr<T>(), tensor_y->dptr<T>(),
                                            ctx->device_ctx()->cuda_stream()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
OF_DEVICE_FUNC void DoUpdateMaximumGrad(int64_t elem_cnt, const T* dz, const T* x, const T* y,
                                        T* dx, T* dy) {
  XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
    if (x[idx] > y[idx]) {
      dx[idx] = dz[idx];
    } else {
      dy[idx] = dz[idx];
    }
  }
}

template<typename T>
__global__ void MaximumBackwardGpuKernel(int64_t elem_cnt, const T* dz, const T* x, const T* y,
                                         T* dx, T* dy) {
  DoUpdateMaximumGrad<T>(elem_cnt, dz, x, y, dx, dy);
}

template<typename T>
class GpuElementwiseMaximumBackwardKernel final : public user_op::OpKernel {
 public:
  GpuElementwiseMaximumBackwardKernel() = default;
  ~GpuElementwiseMaximumBackwardKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const T* dptr_dz = tensor_dz->dptr<T>();
    const T* dptr_x = tensor_x->dptr<T>();
    const T* dptr_y = tensor_y->dptr<T>();

    T* dptr_dx = tensor_dx->mut_dptr<T>();
    T* dptr_dy = tensor_dy->mut_dptr<T>();

    const int cnt = tensor_dz->shape().elem_cnt();
    RUN_CUDA_KERNEL((MaximumBackwardGpuKernel<T>), ctx->device_ctx(), cnt, cnt, dptr_dz, dptr_x,
                    dptr_y, dptr_dx, dptr_dy);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MAXIMUM_GPU_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("elementwise_maximum")                                        \
      .SetCreateFn<GpuElementwiseMaximumKernel<dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                 \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_MAXIMUM_GPU_KERNEL(float);
REGISTER_MAXIMUM_GPU_KERNEL(double);

#define REGISTER_BW_MAXIMUM_GPU_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("elementwise_maximum_backward")               \
      .SetCreateFn<GpuElementwiseMaximumBackwardKernel<dtype>>()     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU) \
                       & (user_op::HobDataType("dz", 0) == GetDataType<dtype>::value));

REGISTER_BW_MAXIMUM_GPU_KERNEL(float);
REGISTER_BW_MAXIMUM_GPU_KERNEL(double);
}  // namespace user_op
}  // namespace oneflow
#endif  // WITH_CUDA
