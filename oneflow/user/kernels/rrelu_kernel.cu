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
#include "oneflow/user/kernels/distributions/distribution_template_util.cuh"
#include "oneflow/user/kernels/distributions/common.h"
namespace oneflow {
namespace {

template<typename T, int unroll_factor, typename F>
__global__ void compute_rrelu(const T* in, T* out, T* noise_data, int64_t elem_cnt, const T lower,
                              const T upper, uint64_t seed, uint64_t offset, const F& random_func) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, id, offset, &state);
  int grid_stride = blockDim.x * gridDim.x * unroll_factor;
  int rounded_size = ((elem_cnt - 1) / grid_stride + 1) * grid_stride;
  T range = upper - lower;
  for (int linear_index = id; linear_index < rounded_size; linear_index += grid_stride) {
    auto rand = random_func(&state);

    // ensure that (&rand.x)[ii] is safe
    static_assert(sizeof(rand) / sizeof(rand.x) == unroll_factor, "");
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li >= elem_cnt) { continue; }
      T r = static_cast<T>((&rand.x)[ii]);
      r = r * range + lower;
      if (in[li] <= static_cast<T>(0)) {
        out[li] = in[li] * r;
        noise_data[li] = r;
      } else {
        out[li] = in[li];
        noise_data[li] = static_cast<T>(1);
      }
    }
    __syncthreads();
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

    if (std::is_same<T, double>::value) {
      DistributionFunctor<DistributionOp::kUniform2Double> dist_functor;
      compute_rrelu<T, 2,decltype(dist_functor)><<<grid, block, 0, cuda_stream->cuda_stream()>>>(
          in_ptr, out_ptr, noise_ptr, size, lower, upper, seed, offset, dist_functor);

    } else {
      // float
       DistributionFunctor<DistributionOp::kUniform4> dist_functor;
      compute_rrelu<T, 4,decltype(dist_functor)><<<grid, block, 0, cuda_stream->cuda_stream()>>>(
          in_ptr, out_ptr, noise_ptr, size, lower, upper, seed, offset, dist_functor);
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