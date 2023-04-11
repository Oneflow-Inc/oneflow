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

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"
#include "oneflow/user/kernels/arange_kernel_util.h"
#include "oneflow/user/kernels/radix_sort.cuh"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/distributions/distribution_template_util.cuh"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {
__global__ void GeneKeysAndValues(const int32_t n, uint64_t seed, uint64_t offset, int32_t* values,
                                  int32_t* keys) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, id, offset, &state);
  CUDA_1D_KERNEL_LOOP(i, n) {
    keys[i] = curand(&state);
    values[i] = i;
  }
}

__global__ void tempcopy2output(const int32_t n, const int32_t offset, int32_t* temp,
                                int32_t* output) {
  CUDA_1D_KERNEL_LOOP(i, n) { output[i] = temp[offset + i]; }
}
class GpuRandPermKernelCache final : public user_op::OpKernelCache {
 public:
  GpuRandPermKernelCache(int32_t lower, int32_t upper) : lower_(lower), upper_(upper) {}
  ~GpuRandPermKernelCache() override = default;

  int32_t lower() const { return lower_; }
  int32_t upper() const { return upper_; }

 private:
  const int32_t lower_;
  const int32_t upper_;
};

namespace {

template<typename K>
size_t GetCubSortPairsTempStorageSize(int64_t n) {
  size_t cub_sort_temp_store_size = 0;
  OF_CUDA_CHECK((cub::DeviceRadixSort::SortPairs<K, K>(nullptr, cub_sort_temp_store_size, nullptr,
                                                       nullptr, nullptr, nullptr, n)));
  size_t temp_store_size = GetCudaAlignedSize(cub_sort_temp_store_size);
  CHECK_GE(temp_store_size, 0) << "temp_store_size should >= 0.";
  CHECK_LT(temp_store_size, static_cast<size_t>(GetMaxVal<int64_t>()))
      << "temp_store_size should < " << static_cast<size_t>(GetMaxVal<int64_t>());
  return temp_store_size;
}

}  // namespace

class GpuRandPermKernel final : public user_op::OpKernel {
 public:
  GpuRandPermKernel() = default;
  ~GpuRandPermKernel() = default;
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    if (parallel_num > 1) {
      const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
      const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
      int64_t parallel_id = ctx->parallel_ctx().parallel_id();
      int32_t n = ctx->Attr<int32_t>("n");
      const Shape& logical_shape = Shape({n});
      TensorSliceView view =
          GetTensorSliceView4ParallelId(hierarchy, nd_sbp, logical_shape, parallel_id);
      std::shared_ptr<GpuRandPermKernelCache> cache(
          new GpuRandPermKernelCache(view.At(0).begin(), view.At(0).end()));
      return cache;
    } else {
      return nullptr;
    }
  }
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(kCUDA));
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int32_t* output = out->mut_dptr<int32_t>();
    const int32_t n = ctx->Attr<int32_t>("n");
    if (n == 0) { return; }
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    auto* stream = ctx->stream();
    const auto device_index = stream->device()->device_index();
    const auto& gpu_generator = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));

    ep::CudaStream* cuda_stream = stream->As<ep::CudaStream>();
    auto execution_policy = gpu_generator->CalcExecutionPolicy(n, cuda_stream);

    auto counter_offset = std::get<0>(execution_policy);
    auto grid = std::get<1>(execution_policy);
    auto block = std::get<2>(execution_policy);

    uint64_t seed = gpu_generator->current_seed();
    uint64_t offset = gpu_generator->get_philox_offset(counter_offset);

    // layout for tmp |...key(in and out,2xN)..|....value....|.... space for sort function....|
    // values are the desired indexes ,and keys are generated randomly.
    void* tmp = tmp_buffer->mut_dptr<void>();
    int32_t* key_base = reinterpret_cast<int32_t*>(tmp);

    const int32_t key_aligned_bytes = GetCudaAlignedSize(n * sizeof(int32_t));
    int32_t* value_base =
        reinterpret_cast<int32_t*>(reinterpret_cast<char*>(key_base) + 2 * key_aligned_bytes);
    const int32_t indices_aligned_bytes = GetCudaAlignedSize(n * sizeof(int32_t));
    int32_t* temp_buffer_base =
        reinterpret_cast<int32_t*>(reinterpret_cast<char*>(value_base) + indices_aligned_bytes);
    const int32_t temp_buffer_aligned_bytes = GetCudaAlignedSize(n * sizeof(int32_t));

    void* tmp_base = reinterpret_cast<void*>(reinterpret_cast<char*>(temp_buffer_base)
                                             + temp_buffer_aligned_bytes);
    size_t temp_storage_bytes = GetCubSortPairsTempStorageSize<int32_t>(n);
    GeneKeysAndValues<<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        n, seed, offset, value_base, key_base);
    if (cache == nullptr) {
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
    } else {
      auto err = cub::DeviceRadixSort::SortPairs(
          /* d_temp_storage */ tmp_base,
          /* temp_storage_bytes */ temp_storage_bytes,
          /* d_keys_in */ key_base,
          /* d_keys_out */ key_base + n,
          /* d_values_in */ value_base,
          /* d_values_out */ temp_buffer_base,
          /* num_items */ n,
          /* begin_bit */ 0,
          /* end_bit */ sizeof(int32_t) * 8,
          /* stream */ ctx->stream()->As<ep::CudaStream>()->cuda_stream());
      OF_CUDA_CHECK(err);
      const auto* randperm_cache = dynamic_cast<const GpuRandPermKernelCache*>(cache);
      auto len = randperm_cache->upper() - randperm_cache->lower();
      const int64_t offset = randperm_cache->lower();
      int32_t block_num = gpu_generator->max_block_num();
      tempcopy2output<<<block_num, kCudaThreadsNumPerBlock, 0,
                        ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          len, offset, temp_buffer_base, output);
    }
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
      const int32_t temp_aligned_bytes = GetCudaAlignedSize(n * sizeof(int32_t));

      /* CUB Temp Storage */
      const int32_t temp_storage_bytes = GetCubSortPairsTempStorageSize<int32_t>(n);

      return sorted_in_aligned_bytes + indices_aligned_bytes + temp_storage_bytes
             + temp_aligned_bytes;
    });

}  // namespace oneflow
