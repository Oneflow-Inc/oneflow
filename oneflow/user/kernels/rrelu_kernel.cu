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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/distributions/normal_distribution.h"
#include "oneflow/user/kernels/distributions/distribution_template_util.cuh"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

namespace {

template<typename T, typename ComputeType>
struct UniformTransformFunctor {
  UniformTransformFunctor(ComputeType range, ComputeType lower) : range(range), lower(lower) {}
  __device__ T operator()(ComputeType random_val) const {
    return static_cast<T>(random_val * range + lower);
  }
  ComputeType range;
  ComputeType lower;
};

template<typename T, typename ComputeType, int unroll_factor, typename Distribution,
         typename Transform>
OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__
    void RReluKernel(int64_t numel, uint64_t seed, uint64_t offset, const T* in_ptr, T* out_ptr,
                     T* noise_data_ptr, Distribution dist_func, Transform transform_func) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  int rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) * blockDim.x
                     * gridDim.x * unroll_factor;
  for (int32_t linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    auto rand = dist_func(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        T r = transform_func(static_cast<ComputeType>((&rand.x)[ii]));
        if (in_ptr[li] <= static_cast<T>(0)) {
          out_ptr[li] = in_ptr[li] * r;
          noise_data_ptr[li] = r;
        } else {
          out_ptr[li] = in_ptr[li];
          noise_data_ptr[li] = static_cast<T>(1);
        }
      }
    }
  }
}

}  // namespace

template<typename T>
class CudaRReluKernel final : public user_op::OpKernel {
 public:
  CudaRReluKernel() = default;
  ~CudaRReluKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA));
    generator->set_current_seed(CHECK_JUST(
        GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"), {"output", 0})));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const int64_t size = in->shape_view().elem_cnt();
    if (size == 0) return;

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* noise_data = ctx->Tensor4ArgNameAndIndex("noise_data", 0);
    const T& lower = ctx->Attr<float>("lower");
    const T& upper = ctx->Attr<float>("upper");

    T* out_ptr = out->mut_dptr<T>();
    T* noise_ptr = noise_data->mut_dptr<T>();
    const T* in_ptr = in->dptr<T>();

    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    ep::CudaStream* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const auto device_index = ctx->stream()->device()->device_index();
    std::shared_ptr<one::CUDAGeneratorImpl> cuda_gen =
        CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));
    auto execution_policy = cuda_gen->CalcExecutionPolicy(size, cuda_stream);
    auto counter_offset = std::get<0>(execution_policy);
    uint64_t seed = cuda_gen->current_seed();
    uint64_t offset = cuda_gen->get_philox_offset(counter_offset);

    auto grid = std::get<1>(execution_policy);
    auto block = std::get<2>(execution_policy);

    using ComputeType = typename distribution::DefaultComputeType<T>::type;
    UniformTransformFunctor<T, ComputeType> transform_functor(
        static_cast<ComputeType>(upper - lower), static_cast<ComputeType>(lower));
    if (std::is_same<T, double>::value) {
      DistributionFunctor<DistributionOp::kUniform2Double> dist_functor;
      RReluKernel<T, ComputeType, 2, decltype(dist_functor), decltype(transform_functor)>
          <<<grid, block, 0, cuda_stream->cuda_stream()>>>(
              size, seed, offset, in_ptr, out_ptr, noise_ptr, dist_functor, transform_functor);
    } else {
      // float
      DistributionFunctor<DistributionOp::kUniform4> dist_functor;
      RReluKernel<T, ComputeType, 4, decltype(dist_functor), decltype(transform_functor)>
          <<<grid, block, 0, cuda_stream->cuda_stream()>>>(
              size, seed, offset, in_ptr, out_ptr, noise_ptr, dist_functor, transform_functor);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_CUDA_RRELU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("rrelu").SetCreateFn<CudaRReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                  \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_RRELU_KERNEL(float)
REGISTER_CUDA_RRELU_KERNEL(double)

}  // namespace oneflow
