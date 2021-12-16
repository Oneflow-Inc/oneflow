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

// data partition: nspace|size|cod
// total thread number: nspace * cod
// in cod part, use cod threads to calculate as follows(m=cod-1, n=dim-1):
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
__global__ void CumsumForwardGpu(const T* pin, T* pout, int64_t nspace, int64_t size, int64_t cod) {
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x;
       i < nspace * cod; i += step) {
    auto space_id = i / cod;
    auto space_thread_id = i % cod;

    auto* pin_base = pin + space_id * size * cod + space_thread_id;
    auto* pout_base = pout + space_id * size * cod + space_thread_id;

    // calculate size data in one thread
    for (auto j = 0; j < size; j++) {
      auto idx = j * cod;
      pout_base[idx] = pin_base[idx];
      if (j != 0) { pout_base[idx] += pout_base[idx - cod]; }
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
    auto nele = in->shape().elem_cnt();
    if (!nele) { return; }

    auto* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* pin = in->dptr<T>();
    auto* pout = out->mut_dptr<T>();

    // size means dimension size, cod means coefficient of dimension
    auto size = in->shape().At(dim);
    auto space = in->shape().Count(dim);
    auto cod = in->shape().Count(dim) / size;
    auto nspace = nele / space;
    auto thread_num = nspace * cod;

    RUN_CUDA_KERNEL((CumsumForwardGpu<T>), ctx->stream(), thread_num, pin, pout, nspace, size, cod);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_CUMSUM_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<GpuCumsumKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                   \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CUMSUM_KERNEL(float)
REGISTER_CUDA_CUMSUM_KERNEL(double)

}  // namespace oneflow
