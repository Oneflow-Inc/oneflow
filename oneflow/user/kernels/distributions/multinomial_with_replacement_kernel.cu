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
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"

// NOTE(Liang Depeng): The implementation of MultinomialWithReplacementGpuKernel is modified from
//                    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/MultinomialKernel.cu#L324
namespace oneflow {

namespace {

template<typename T>
__device__ int binarySearchForMultinomial(const T* cumdist, const T* dist, int32_t size, T val) {
  int start = 0;
  int end = size;

  while (end - start > 0) {
    int mid = start + (end - start) / 2;
    T midVal = cumdist[mid];
    if (midVal < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first non-zero element by setting start to size-1 here,
    // the code below will move it to the last non-zero probability
    // this actually can happen when the random number is 1
    // (github pytorch issue #4858).
    start = size - 1;
  }

  while (start >= 1 && dist[start] == 0) start--;

  return start;
}

template<typename T>
__global__ void sampleMultinomialWithReplacement(uint64_t seed, uint64_t offset,
                                                 int32_t totalSamples, int64_t* dest,
                                                 int64_t distributions, int64_t categories,
                                                 const T* normDistPrefixSum, const T* normDist) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on.

  // global index formula for 2D grid of 1D blocks
  int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  // The block determines the distribution for which we generate a point
  for (int64_t curDist = blockIdx.y; curDist < distributions; curDist += gridDim.y) {
    for (int sample = blockIdx.x * blockDim.x + threadIdx.x; sample < totalSamples;
         sample += blockDim.x * gridDim.x) {
      // we are losing 3 out of 4 generated numbers but it's ok
      // this kernel is not very efficient anyway
      auto rand = curand_uniform4(&state);
      T r = static_cast<T>(rand.x);

      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial<T>(normDistPrefixSum + curDist * categories,
                                                 normDist + curDist * categories, categories, r);

      dest[curDist * totalSamples + sample] = choice;
    }
  }
}

}  // namespace

template<typename T>
class MultinomialWithReplacementGpuKernel final : public user_op::OpKernel {
 public:
  MultinomialWithReplacementGpuKernel() = default;
  ~MultinomialWithReplacementGpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA));
    // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    auto gpu_gen = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());

    const user_op::Tensor* norm_dist = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* prefix_sum = ctx->Tensor4ArgNameAndIndex("prefix_sum", 0);
    CHECK_NOTNULL(prefix_sum);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const T* norm_dist_ptr = norm_dist->dptr<T>();
    const T* prefix_sum_ptr = prefix_sum->dptr<T>();
    int64_t* result_ptr = out->mut_dptr<int64_t>();

    int64_t numCategories = norm_dist->shape_view().At(norm_dist->shape_view().NumAxes() - 1);
    int64_t numDist = norm_dist->shape_view().NumAxes() > 1 ? norm_dist->shape_view().At(0) : 1;
    const int32_t n_sample = ctx->Attr<int32_t>("num_samples");

    // Binary search is warp divergent (so effectively we're running
    // with just a single thread), but for better utilization,
    // we need each block to have at least 4 warps.
    dim3 block(128);

    ep::CudaStream* stream = ctx->stream()->As<ep::CudaStream>();
    // Each block will generate a sample from one
    // distribution concurrently.
    int grid_y = std::min<int>(numDist, stream->device_properties().maxGridSize[1]);
    dim3 grid((n_sample - 1) / block.x + 1, grid_y);
    uint64_t seed = gpu_gen->current_seed();
    uint64_t offset = gpu_gen->get_philox_offset(((numDist - 1) / grid.y + 1) * 4);

    // Sample with replacement
    sampleMultinomialWithReplacement<<<grid, block, 0, stream->cuda_stream()>>>(
        seed, offset, n_sample, result_ptr, numDist, numCategories, prefix_sum_ptr, norm_dist_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MULTINOMIAL_WITH_REPLACEMENT_GPU_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("multinomial_with_replacement")                                \
      .SetCreateFn<MultinomialWithReplacementGpuKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("prefix_sum", 0) == GetDataType<dtype>::value));

REGISTER_MULTINOMIAL_WITH_REPLACEMENT_GPU_KERNEL(float)
REGISTER_MULTINOMIAL_WITH_REPLACEMENT_GPU_KERNEL(double)

}  // namespace oneflow
