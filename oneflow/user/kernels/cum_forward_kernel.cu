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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {
#ifdef WITH_CUDA
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
template<typename T, template<typename> class BinaryFunc>
__global__ void CumsumForwardGpu(const T* in_ptr, T* out_ptr, int64_t cs_up_space, int64_t cs_space,
                                 int64_t cs_down_space) {
  CUDA_1D_KERNEL_LOOP(i, cs_up_space * cs_down_space) {
    auto cs_up_space_id = i / cs_down_space;
    auto cs_down_space_id = i - (i / cs_down_space) * cs_down_space;

    auto* in_ptr_base = in_ptr + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;
    auto* out_ptr_base = out_ptr + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;

    // calculate cs_space data in one thread
    for (auto j = 0; j < cs_space; j++) {
      auto idx = j * cs_down_space;
      out_ptr_base[idx] = in_ptr_base[idx];
      if (j != 0) {
        out_ptr_base[idx] =
            BinaryFunc<T>::Invoke(out_ptr_base[idx], out_ptr_base[idx - cs_down_space]);
      }
    }
  }
}
template<typename T, template<typename> class BinaryFunc>
__global__ void CumsumForwardGpuUpSpaceIs1(const T* in_ptr, T* out_ptr, int64_t cs_space,
                                           int64_t cs_down_space) {
  CUDA_1D_KERNEL_LOOP(i, cs_down_space) {
    auto* in_ptr_base = in_ptr + i;
    auto* out_ptr_base = out_ptr + i;

    // calculate cs_space data in one thread
    for (auto j = 0; j < cs_space; j++) {
      auto idx = j * cs_down_space;
      out_ptr_base[idx] = in_ptr_base[idx];
      if (j != 0) {
        out_ptr_base[idx] =
            BinaryFunc<T>::Invoke(out_ptr_base[idx], out_ptr_base[idx - cs_down_space]);
      }
    }
  }
}
template<typename T, template<typename> class BinaryFunc>
__global__ void CumsumForwardGpuDownSpaceIs1(const T* in_ptr, T* out_ptr, int64_t cs_up_space,
                                             int64_t cs_space) {
  CUDA_1D_KERNEL_LOOP(i, cs_up_space) {
    auto* in_ptr_base = in_ptr + i * cs_space;
    auto* out_ptr_base = out_ptr + i * cs_space;

    // calculate cs_space data in one thread
    for (auto j = 0; j < cs_space; j++) {
      out_ptr_base[j] = in_ptr_base[j];
      if (j != 0) { out_ptr_base[j] = BinaryFunc<T>::Invoke(out_ptr_base[j], out_ptr_base[j - 1]); }
    }
  }
}
}  // namespace

template<typename T, template<typename> class BinaryFunc>
class GpuCumKernel : public user_op::OpKernel {
 public:
  GpuCumKernel() = default;
  ~GpuCumKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // judge whether tensor has 0 size dimension first
    const auto* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto elem_cnt = in->shape().elem_cnt();
    if (!elem_cnt) { return; }

    auto* out = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* in_ptr = in->dptr<T>();
    auto* out_ptr = out->mut_dptr<T>();

    // data partition: up_space|space|down_space
    auto up_space = elem_cnt / in->shape().Count(dim);
    auto space = in->shape().At(dim);
    auto down_space = in->shape().Count(dim + 1);
    auto thread_num = up_space * down_space;

    if (up_space == 1) {
      RUN_CUDA_KERNEL((CumsumForwardGpuUpSpaceIs1<T, BinaryFunc>), ctx->stream(), thread_num,
                      in_ptr, out_ptr, space, down_space);
    } else if (down_space == 1) {
      RUN_CUDA_KERNEL((CumsumForwardGpuDownSpaceIs1<T, BinaryFunc>), ctx->stream(), thread_num,
                      in_ptr, out_ptr, up_space, space);
    } else {
      RUN_CUDA_KERNEL((CumsumForwardGpu<T, BinaryFunc>), ctx->stream(), thread_num, in_ptr, out_ptr,
                      up_space, space, down_space);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class GpuCumSumKernel final : public GpuCumKernel<T, BinaryFuncAdd> {
 public:
  GpuCumSumKernel() = default;
  ~GpuCumSumKernel() = default;
};

#define REGISTER_CUDA_CUMSUM_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<GpuCumSumKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                   \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CUMSUM_KERNEL(int32_t)
REGISTER_CUDA_CUMSUM_KERNEL(int64_t)
REGISTER_CUDA_CUMSUM_KERNEL(float)
REGISTER_CUDA_CUMSUM_KERNEL(double)
#undef REGISTER_CUDA_CUMSUM_KERNEL

template<typename T>
class GpuCumProdKernel final : public GpuCumKernel<T, BinaryFuncMul> {
 public:
  GpuCumProdKernel() = default;
  ~GpuCumProdKernel() = default;
};

#define REGISTER_CUDA_CUMPROD_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("cumprod").SetCreateFn<GpuCumProdKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                     \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CUMPROD_KERNEL(int32_t)
REGISTER_CUDA_CUMPROD_KERNEL(int64_t)
REGISTER_CUDA_CUMPROD_KERNEL(float)
REGISTER_CUDA_CUMPROD_KERNEL(double)
#undef REGISTER_CUDA_CUMPROD_KERNEL
#endif
}  // namespace oneflow
