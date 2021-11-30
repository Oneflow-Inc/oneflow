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
#include <curand.h>
#include <curand_kernel.h>

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"
#include "oneflow/user/kernels/arange_kernel_util.h"
#include "oneflow/user/kernels/radix_sort.cuh"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
__global__ void GeneKeysAndValues(const int32_t n, int32_t* values, int32_t* keys,
                                  curandState* state) {
  XPU_1D_KERNEL_LOOP(i, n) {
    keys[i] = curand(state + i);
    values[i] = i;
  }
}

class GpuRandPermKernel final : public user_op::OpKernel {
 public:
  GpuRandPermKernel() = default;
  ~GpuRandPermKernel() = default;
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(kCUDA));
    generator->set_current_seed(ctx->Attr<int64_t>("seed"));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int32_t* output = out->mut_dptr<int32_t>();
    const int32_t n = ctx->Attr<int32_t>("n");
    if (n == 0) { return; }
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    const auto& gpu_generator = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());
    CHECK_NOTNULL(generator);

    int32_t block_num = gpu_generator->max_block_num();
    int32_t thread_num = gpu_generator->max_thread_num();
    curandState* curand_states = gpu_generator->curand_states();

    // layout for tmp |...key(in and out,2xN)..|....value....|.... space for sort function....|
    // values are the desired indexes ,and keys are generated randomly.
    void* tmp = tmp_buffer->mut_dptr<void>();
    int32_t* key_base = reinterpret_cast<int32_t*>(tmp);

    const int32_t key_aligned_bytes = GetCudaAlignedSize(n * sizeof(int32_t));
    int32_t* value_base =
        reinterpret_cast<int32_t*>(reinterpret_cast<char*>(key_base) + 2 * key_aligned_bytes);

    const int32_t indices_aligned_bytes = GetCudaAlignedSize(n * sizeof(int32_t));
    void* tmp_base =
        reinterpret_cast<void*>(reinterpret_cast<char*>(value_base) + indices_aligned_bytes);
    size_t temp_storage_bytes = InferTempStorageForSortPairsDescending<int32_t, int32_t>(1, n);

    GeneKeysAndValues<<<block_num, kCudaThreadsNumPerBlock, 0,
                        ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        n, value_base, key_base, curand_states);

    auto err = cub::DeviceRadixSort::SortPairs(
        /* d_temp_storage */ tmp_base,
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_keys_in */ key_base,
        /* d_keys_out */ key_base + n,
        /* d_values_in */ value_base,
        /* d_values_out */ output,
        /* num_items */ n,
        /* begin_bit */ 0,
        /* end_bit */ sizeof(int32_t) * 8,
        /* stream */ ctx->stream()->As<ep::CudaStream>()->cuda_stream());
    OF_CUDA_CHECK(err);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
REGISTER_USER_KERNEL("randperm")
    .SetCreateFn<GpuRandPermKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      const int32_t n = ctx->Attr<int32_t>("n");
      /* Sorted In */
      const int32_t sorted_in_aligned_bytes = 2 * GetCudaAlignedSize(n * sizeof(int32_t));
      /* Indices */
      const int32_t indices_aligned_bytes = GetCudaAlignedSize(n * sizeof(int32_t));

      /* CUB Temp Storage */
      const int32_t temp_storage_bytes =
          InferTempStorageForSortPairsDescending<int32_t, int32_t>(1, n);

      return sorted_in_aligned_bytes + indices_aligned_bytes + temp_storage_bytes;
    });

}  // namespace oneflow
