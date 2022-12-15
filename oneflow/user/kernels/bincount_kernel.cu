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
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/user_op_hob.h"
#include "oneflow/core/ep/include/primitive/memset.h"

namespace oneflow {
namespace user_op {
namespace {

// clang-format off
template<typename IDX, typename T>
__global__ static void BinCountCompute(const IDX* in_ptr, const T* weight, T* out_ptr, int64_t size) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    IDX idx = *(in_ptr + i);
    cuda::atomic::Add(out_ptr + idx, weight[i]);
  }
};
// clang-format on

template<typename IDX, typename T>
__global__ static void BinCountCompute(const IDX* in_ptr, T* out_ptr, int64_t size) {
  T one = GetOneVal<T>();
  CUDA_1D_KERNEL_LOOP(i, size) {
    IDX idx = *(in_ptr + i);
    cuda::atomic::Add(out_ptr + idx, one);
  }
};

template<typename IDX, typename T>
class CUDABinCountKernel final : public user_op::OpKernel {
 public:
  CUDABinCountKernel() = default;
  ~CUDABinCountKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    size_t out_size = ctx->Attr<int64_t>("size") * sizeof(T);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const IDX* in_ptr = in->dptr<IDX>();
    T* out_ptr = out->mut_dptr<T>();
    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), out_ptr, 0, out_size);
    int64_t in_size = in->shape_view().elem_cnt();
    if (in_size == 0) { return; }
    if (ctx->has_input("weight", 0)) {
      const T* weight_ptr = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>();
      BinCountCompute<IDX, T><<<BlocksNum4ThreadsNum(in_size), kCudaThreadsNumPerBlock, 0,
                                ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          in_ptr, weight_ptr, out_ptr, in_size);
    } else {
      BinCountCompute<IDX, T>
          <<<BlocksNum4ThreadsNum(in_size), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(in_ptr, out_ptr, in_size);
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace oneflow

#define REGISTER_CUDA_BINCOUNT_KERNEL(idx_type, dtype)                                    \
  REGISTER_USER_KERNEL("bincount")                                                        \
      .SetCreateFn<CUDABinCountKernel<idx_type, dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                    \
                       && (user_op::HobDataType("in", 0) == GetDataType<idx_type>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_BINCOUNT_KERNEL(int64_t, int64_t)
REGISTER_CUDA_BINCOUNT_KERNEL(int64_t, half)
REGISTER_CUDA_BINCOUNT_KERNEL(int64_t, float)
REGISTER_CUDA_BINCOUNT_KERNEL(int64_t, double)

}  // namespace user_op
}  // namespace oneflow
