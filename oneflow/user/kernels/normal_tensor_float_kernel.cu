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

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
namespace oneflow {

template<typename T, typename ComputeType>
struct NormalTensorFloatFunctor {
  // mean + output * std
  NormalTensorFloatFunctor(ComputeType mean, ComputeType std) : mean(mean), std(std) {}
  OF_DEVICE_FUNC T operator()(ComputeType random_val)(ComputeType random_val) const {
    T output = static_cast<T>(random_val * std + mean);
  }
  ComputeType mean;
  ComputeType std;
};

template<typename T>
class GpuNormalTensorFloatKernel final : public user_op::OpKernel {
 public:
  GpuNormalTensorFloatKernel() = default;
  ~GpuNormalTensorFloatKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const double std = ctx->Attr<double>("std");
    // Use CUDA Elementwise Template.
    OF_CUDA_CHECK((cuda::elementwise::Unary(
        NormalTensorFloatFunctor<T, T>(), elem_cnt, y->mut_dptr<T>(), x->dptr<T>(),
        ctx->stream()->As<ep::CudaStream>()->cuda_stream()
            >>> (elem_cnt, seed, offset, dptr, dist_functor, transform_functor))));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_NORMAL_TENSOR_FLOAT_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("frac").SetCreateFn<GpuNormalTensorFloatKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                            \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_NORMAL_TENSOR_FLOAT_KERNEL(float)
REGISTER_GPU_NORMAL_TENSOR_FLOAT_KERNEL(double)

}  // namespace oneflow
