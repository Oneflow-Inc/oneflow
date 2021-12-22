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
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {

// total thread number: cs_up_space * cs_down_space
// in cs_down_space part, use cs_down_space threads
// to calculate as follows(m=cs_down_space-1, n=cs_space-1, '|' stands for dependency):
// dm0, ..., d10, d00
//  |         |    |
// dm1, ..., d11, d01
//  |         |    |
// dm2, ..., d12, d02
//  |         |    |
// ...       ...  ...
//  |         |    |
// dmn, ..., d1n, d0n
template<typename T>
__global__ void CumsumForwardGpu(const T* pin, T* pout, int64_t cs_up_space, int64_t cs_space,
                                 int64_t cs_down_space) {
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x;
       i < cs_up_space * cs_down_space; i += step) {
    auto cs_up_space_id = i / cs_down_space;
    auto cs_down_space_id = i % cs_down_space;

    auto* pin_base = pin + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;
    auto* pout_base = pout + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;

    // calculate cs_space data in one thread
    for (auto j = 0; j < cs_space; j++) {
      auto idx = j * cs_down_space;
      pout_base[idx] = pin_base[idx];
      if (j != 0) { pout_base[idx] += pout_base[idx - cs_down_space]; }
    }
  }
}

// total thread number: cs_up_space * cs_down_space
// in cs_down_space part, use cs_down_space threads
// to calculate as follows(m=cs_down_space-1, n=cs_space-1, there is no dependency in backward):
// dm0, ..., d10, d00
// dm1, ..., d11, d01
// dm2, ..., d12, d02
// ...       ...  ...
// dmn, ..., d1n, d0n
template<typename T>
__global__ void CumsumBackwardGpu(const T* pin, T* pout, int64_t cs_up_space, int64_t cs_space,
                                  int64_t cs_down_space, int64_t elem_cnt) {
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < elem_cnt;
       i += step) {
    auto cs_space_id = (i % (cs_space * cs_down_space)) / cs_down_space;
    pout[i] = (cs_space - cs_space_id) * pin[i];
  }
}

template<typename T>
__global__ void CumsumBackwardGpu3D(const T* pin, T* pout, int64_t cs_up_space, int64_t cs_space,
                                    int64_t cs_down_space, int64_t elem_cnt) {
  for (auto i = blockIdx.z * blockDim.z + threadIdx.z; i < cs_up_space; i++) {
    for (auto j = blockIdx.y * blockDim.y + threadIdx.y; j < cs_space; j++) {
      for (auto k = blockIdx.x * blockDim.x + threadIdx.x; k < cs_down_space; k++) {
        auto idx = i * cs_space * cs_down_space + j * cs_down_space + k;
        pout[idx] = (cs_space - j) * pin[idx];
      }
    }
  }
}
}  // namespace

template<typename T>
class GpuCumsumKernel final : public user_op::OpKernel {
 public:
  GpuCumsumKernel() = default;
  ~GpuCumsumKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // judge whether tensor has 0 size dimension first
    const auto* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    auto elem_cnt = in->shape().elem_cnt();
    if (!elem_cnt) { return; }

    auto* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* pin = in->dptr<T>();
    auto* pout = out->mut_dptr<T>();

    // take cumsum's abbreviation as `cs`
    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = elem_cnt / in->shape().Count(dim);
    auto cs_space = in->shape().At(dim);
    auto cs_down_space = in->shape().Count(dim) / cs_space;

    auto thread_num = cs_up_space * cs_down_space;
    RUN_CUDA_KERNEL((CumsumForwardGpu<T>), ctx->stream(), thread_num, pin, pout, cs_up_space,
                    cs_space, cs_down_space);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_CUMSUM_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<GpuCumsumKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                   \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CUMSUM_KERNEL(float)
REGISTER_CUDA_CUMSUM_KERNEL(double)

template<typename T>
class GpuCumsumGradKernel final : public user_op::OpKernel {
 public:
  GpuCumsumGradKernel() = default;
  ~GpuCumsumGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // judge whether tensor has 0 size dimension first
    const auto* in = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto elem_cnt = in->shape().elem_cnt();
    if (!elem_cnt) { return; }
    auto* out = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* pin = in->dptr<T>();
    auto* pout = out->mut_dptr<T>();

    // take cumsum's abbreviation as `cs`
    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = elem_cnt / in->shape().Count(dim);
    auto cs_space = in->shape().At(dim);
    auto cs_down_space = in->shape().Count(dim) / cs_space;

    auto thread_num = elem_cnt;
    RUN_CUDA_KERNEL((CumsumBackwardGpu<T>), ctx->stream(), thread_num, pin, pout, cs_up_space,
                    cs_space, cs_down_space, elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_CUMSUM_GRAD_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("cumsum_grad")                                  \
      .SetCreateFn<GpuCumsumGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CUMSUM_GRAD_KERNEL(float)
REGISTER_CUDA_CUMSUM_GRAD_KERNEL(double)

}  // namespace oneflow
