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
#include <cstdint>
#include <memory>
#include "oneflow/user/kernels/dropout_kernel.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

using H2PackType = typename std::aligned_storage<4 * sizeof(half), 4 * sizeof(half)>::type;
union H2Pack {
  H2PackType storage;
  half2 h2[2];
};

template<typename T, int pack_size>
__global__ void MaskAndScaleGpu(uint64_t seed, int32_t* counter, uint64_t* offset,
                                uint64_t counter_offset, const int64_t elem_cnt, float rate,
                                float scale, const T* x, int8_t* mask, T* y) {
  int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, thread_id, *offset, &state);
  using LoadT = typename std::aligned_storage<sizeof(T) * pack_size, sizeof(T) * pack_size>::type;
  using MaskT =
      typename std::aligned_storage<sizeof(int8_t) * pack_size, sizeof(int8_t) * pack_size>::type;

  float4 rand_uniform;
  for (int64_t linear_index = thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    rand_uniform = curand_uniform4(&state);
    rand_uniform.x = rand_uniform.x >= rate;
    rand_uniform.y = rand_uniform.y >= rate;
    rand_uniform.z = rand_uniform.z >= rate;
    rand_uniform.w = rand_uniform.w >= rate;
    const LoadT* x_load = reinterpret_cast<const LoadT*>(&x[linear_index]);
    cuda::elementwise::Pack<T, pack_size> x_vec;
    x_vec.storage = *x_load;

    int8_t mask_vec[pack_size];
    T y_vec[pack_size];
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      mask_vec[i] = (&rand_uniform.x)[i] >= rate;
      y_vec[i] = x_vec.elem[i] * mask_vec[i] * scale;
    }

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }
  __syncthreads();

  if (thread_id == 0) {
    int32_t new_counter = cuda::atomic::Add(counter, 1) + 1;
    if (new_counter == gridDim.x) {
      *counter = 0;               // reset counter to zero
      *offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<typename T, int pack_size>
__global__ void MaskAndScaleAddGpu(uint64_t seed, int32_t* counter, uint64_t* offset,
                                   uint64_t counter_offset, const int64_t elem_cnt, float rate,
                                   float scale, const T* x, int8_t* mask, const T* addend, T* y) {
  int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, thread_id, *offset, &state);
  using LoadT = typename std::aligned_storage<sizeof(T) * pack_size, sizeof(T) * pack_size>::type;
  using MaskT =
      typename std::aligned_storage<sizeof(int8_t) * pack_size, sizeof(int8_t) * pack_size>::type;

  float4 rand_uniform;
  for (int64_t linear_index = thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    rand_uniform = curand_uniform4(&state);
    rand_uniform.x = rand_uniform.x >= rate;
    rand_uniform.y = rand_uniform.y >= rate;
    rand_uniform.z = rand_uniform.z >= rate;
    rand_uniform.w = rand_uniform.w >= rate;
    const LoadT* x_load = reinterpret_cast<const LoadT*>(&x[linear_index]);
    cuda::elementwise::Pack<T, pack_size> x_vec;
    x_vec.storage = *x_load;

    const LoadT* addend_load = reinterpret_cast<const LoadT*>(&addend[linear_index]);
    cuda::elementwise::Pack<T, pack_size> addend_vec;
    addend_vec.storage = *addend_load;

    int8_t mask_vec[pack_size];
    T y_vec[pack_size];
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      mask_vec[i] = (&rand_uniform.x)[i] >= rate;
      y_vec[i] = x_vec.elem[i] * mask_vec[i] * scale + addend_vec.elem[i];
    }

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }
  __syncthreads();

  if (thread_id == 0) {
    int32_t new_counter = cuda::atomic::Add(counter, 1) + 1;
    if (new_counter == gridDim.x) {
      *counter = 0;               // reset counter to zero
      *offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<>
__global__ void MaskAndScaleGpu<half, 4>(uint64_t seed, int32_t* counter, uint64_t* offset,
                                         uint64_t counter_offset, const int64_t elem_cnt,
                                         float rate, float scale, const half* x, int8_t* mask,
                                         half* y) {
  int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, thread_id, *offset, &state);
  using LoadT = typename std::aligned_storage<sizeof(half) * 4, sizeof(half) * 4>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * 4, sizeof(int8_t) * 4>::type;

  float4 rand_uniform;
  half2 h2_scale = __float2half2_rn(scale);
  for (int64_t linear_index = thread_id * 4; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * 4) {
    rand_uniform = curand_uniform4(&state);

    const LoadT* x_load = reinterpret_cast<const LoadT*>(&x[linear_index]);
    H2Pack x_vec{};
    x_vec.storage = *x_load;

    int8_t mask_vec[4];
    half2 y_vec[2];
    half2 one_or_zero_h2[2];

    mask_vec[0] = (&rand_uniform.x)[0] >= rate;
    one_or_zero_h2[0].x = mask_vec[0];
    mask_vec[1] = (&rand_uniform.y)[1] >= rate;
    one_or_zero_h2[0].y = mask_vec[1];
    y_vec[0] = __hmul2(__hmul2(x_vec.h2[0], one_or_zero_h2[0]), h2_scale);

    mask_vec[2] = (&rand_uniform.z)[2] >= rate;
    one_or_zero_h2[1].x = mask_vec[2];
    mask_vec[3] = (&rand_uniform.w)[3] >= rate;
    one_or_zero_h2[1].y = mask_vec[3];
    y_vec[1] = __hmul2(__hmul2(x_vec.h2[1], one_or_zero_h2[1]), h2_scale);

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }
  __syncthreads();

  if (thread_id == 0) {
    int32_t new_counter = cuda::atomic::Add(counter, 1) + 1;
    if (new_counter == gridDim.x) {
      *counter = 0;               // reset counter to zero
      *offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<>
__global__ void MaskAndScaleAddGpu<half, 4>(uint64_t seed, int32_t* counter, uint64_t* offset,
                                            uint64_t counter_offset, const int64_t elem_cnt,
                                            float rate, float scale, const half* x, int8_t* mask,
                                            const half* addend, half* y) {
  int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, thread_id, *offset, &state);
  using LoadT = typename std::aligned_storage<sizeof(half) * 4, sizeof(half) * 4>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * 4, sizeof(int8_t) * 4>::type;

  float4 rand_uniform;
  half2 h2_scale = __float2half2_rn(scale);
  for (int64_t linear_index = thread_id * 4; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * 4) {
    rand_uniform = curand_uniform4(&state);

    const LoadT* x_load = reinterpret_cast<const LoadT*>(&x[linear_index]);
    H2Pack x_vec{};
    x_vec.storage = *x_load;

    const LoadT* addend_load = reinterpret_cast<const LoadT*>(&addend[linear_index]);
    H2Pack addend_vec{};
    addend_vec.storage = *addend_load;

    int8_t mask_vec[4];
    half2 y_vec[2];
    half2 one_or_zero_h2[2];

    mask_vec[0] = (&rand_uniform.x)[0] >= rate;
    one_or_zero_h2[0].x = mask_vec[0];
    mask_vec[1] = (&rand_uniform.y)[1] >= rate;
    one_or_zero_h2[0].y = mask_vec[1];
    y_vec[0] =
        __hadd2(__hmul2(__hmul2(x_vec.h2[0], one_or_zero_h2[0]), h2_scale), addend_vec.h2[0]);

    mask_vec[2] = (&rand_uniform.z)[2] >= rate;
    one_or_zero_h2[1].x = mask_vec[2];
    mask_vec[3] = (&rand_uniform.w)[3] >= rate;
    one_or_zero_h2[1].y = mask_vec[3];
    y_vec[1] =
        __hadd2(__hmul2(__hmul2(x_vec.h2[1], one_or_zero_h2[1]), h2_scale), addend_vec.h2[0]);

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }
  __syncthreads();

  if (thread_id == 0) {
    int32_t new_counter = cuda::atomic::Add(counter, 1) + 1;
    if (new_counter == gridDim.x) {
      *counter = 0;               // reset counter to zero
      *offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<>
__global__ void MaskAndScaleGpu<double, 2>(uint64_t seed, int32_t* counter, uint64_t* offset,
                                           uint64_t counter_offset, const int64_t elem_cnt,
                                           float rate, float scale, const double* x, int8_t* mask,
                                           double* y) {
  int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, thread_id, *offset, &state);
  using LoadT = typename std::aligned_storage<sizeof(double) * 2, sizeof(double) * 2>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * 2, sizeof(int8_t) * 2>::type;

  float4 rand_uniform;
  for (int64_t linear_index = thread_id * 2; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * 2) {
    rand_uniform = curand_uniform4(&state);
    // Generate 4 float and use last 2 element to mask 2 double input elements.
    rand_uniform.x = rand_uniform.z;
    rand_uniform.y = rand_uniform.w;
    rand_uniform.x = rand_uniform.x >= rate;
    rand_uniform.y = rand_uniform.y >= rate;
    const LoadT* x_load = reinterpret_cast<const LoadT*>(&x[linear_index]);
    cuda::elementwise::Pack<double, 2> x_vec;
    x_vec.storage = *x_load;

    int8_t mask_vec[2];
    double y_vec[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      mask_vec[i] = (&rand_uniform.x)[i] >= rate;
      y_vec[i] = x_vec.elem[i] * mask_vec[i] * scale;
    }

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }
  __syncthreads();

  if (thread_id == 0) {
    int32_t new_counter = cuda::atomic::Add(counter, 1) + 1;
    if (new_counter == gridDim.x) {
      *counter = 0;               // reset counter to zero
      *offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<>
__global__ void MaskAndScaleAddGpu<double, 2>(uint64_t seed, int32_t* counter, uint64_t* offset,
                                              uint64_t counter_offset, const int64_t elem_cnt,
                                              float rate, float scale, const double* x,
                                              int8_t* mask, const double* addend, double* y) {
  int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, thread_id, *offset, &state);
  using LoadT = typename std::aligned_storage<sizeof(double) * 2, sizeof(double) * 2>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * 2, sizeof(int8_t) * 2>::type;

  float4 rand_uniform;
  for (int64_t linear_index = thread_id * 2; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * 2) {
    rand_uniform = curand_uniform4(&state);
    rand_uniform.x = rand_uniform.x = rand_uniform.z;
    rand_uniform.y = rand_uniform.x = rand_uniform.w;
    rand_uniform.x = rand_uniform.x >= rate;
    rand_uniform.y = rand_uniform.y >= rate;
    const LoadT* x_load = reinterpret_cast<const LoadT*>(&x[linear_index]);
    cuda::elementwise::Pack<double, 2> x_vec;
    x_vec.storage = *x_load;

    const LoadT* addend_load = reinterpret_cast<const LoadT*>(&addend[linear_index]);
    cuda::elementwise::Pack<double, 2> addend_vec;
    addend_vec.storage = *addend_load;

    int8_t mask_vec[2];
    double y_vec[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      mask_vec[i] = (&rand_uniform.x)[i] >= rate;
      y_vec[i] = x_vec.elem[i] * mask_vec[i] * scale + addend_vec.elem[i];
    }

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }
  __syncthreads();

  if (thread_id == 0) {
    int32_t new_counter = cuda::atomic::Add(counter, 1) + 1;
    if (new_counter == gridDim.x) {
      *counter = 0;               // reset counter to zero
      *offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<int pack_size>
unsigned int ComputeGridSize(const int32_t block_size, const int64_t elem_cnt) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
  unsigned int grid_size = ((elem_cnt + block_size - 1) / block_size);
  grid_size = std::min((unsigned int)prop.multiProcessorCount * blocks_per_sm, grid_size);
  return grid_size;
}

template<typename T>
void MaskAndScale(DeviceCtx* ctx, uint64_t seed, int32_t* counter, uint64_t* offset,
                  const int64_t elem_cnt, float rate, float scale, const T* x, int8_t* mask, T* y) {
  int32_t UNROLL = 4;
  int32_t block_size = 256;
  unsigned int grid_size = ComputeGridSize<4>(block_size, elem_cnt);
  uint64_t counter_offset = ((elem_cnt - 1) / (block_size * grid_size * UNROLL) + 1) * UNROLL;
  MaskAndScaleGpu<T, 4><<<grid_size, block_size, 0, ctx->cuda_stream()>>>(
      seed, counter, offset, counter_offset, elem_cnt, rate, scale, x, mask, y);
}

template<typename T>
void MaskAndScaleAdd(DeviceCtx* ctx, uint64_t seed, int32_t* counter, uint64_t* offset,
                     const int64_t elem_cnt, float rate, float scale, const T* x, int8_t* mask,
                     const T* addend, T* y) {
  int32_t UNROLL = 4;
  int32_t block_size = 256;
  unsigned int grid_size = ComputeGridSize<4>(block_size, elem_cnt);
  uint64_t counter_offset = ((elem_cnt - 1) / (block_size * grid_size * UNROLL) + 1) * UNROLL;
  MaskAndScaleAddGpu<T, 4><<<grid_size, block_size, 0, ctx->cuda_stream()>>>(
      seed, counter, offset, counter_offset, elem_cnt, rate, scale, x, mask, addend, y);
}

template<typename T>
struct MaskAndScaleFunctor {
  OF_DEVICE_FUNC explicit MaskAndScaleFunctor(float scale) : scale(scale) {}
  OF_DEVICE_FUNC T operator()(T x, int8_t mask) const {
    return x * static_cast<T>(mask) * static_cast<T>(scale);
  }
  float scale;
};

template<typename T>
class DropoutKernelGPU final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  DropoutKernelGPU() = default;
  ~DropoutKernelGPU() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    auto* fused_dropout_kernel_state = dynamic_cast<FusedDropoutKernelState*>(state);
    CHECK_NOTNULL(fused_dropout_kernel_state);
    const auto& generator = fused_dropout_kernel_state->generator();
    CHECK_NOTNULL(generator);
    std::shared_ptr<one::CUDAGeneratorImpl> cuda_gen =
        CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());
    uint64_t seed = cuda_gen->current_seed();

    const float rate = ctx->Attr<float>("rate");
    float scale = 1.0;
    if (rate != 1.0) { scale = 1.0 / (1.0 - rate); }
    int32_t* counter = cuda_gen->dev_counter();
    uint64_t* offset = cuda_gen->dev_offset();

    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* addend = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      MaskAndScaleAdd<T>(ctx->device_ctx(), seed, counter, offset, in->shape().elem_cnt(), rate,
                         scale, in->dptr<T>(), mask->mut_dptr<int8_t>(), addend->dptr<T>(),
                         out->mut_dptr<T>());
    } else {
      MaskAndScale<T>(ctx->device_ctx(), seed, counter, offset, in->shape().elem_cnt(), rate, scale,
                      in->dptr<T>(), mask->mut_dptr<int8_t>(), out->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL_GPU(dtype)                                                \
  REGISTER_USER_KERNEL("dropout").SetCreateFn<DropoutKernelGPU<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu")                                                  \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)                     \
      & (user_op::HobDataType("mask", 0) == GetDataType<int8_t>::value));

REGISTER_DROPOUT_KERNEL_GPU(half)
REGISTER_DROPOUT_KERNEL_GPU(float)
REGISTER_DROPOUT_KERNEL_GPU(double)

template<typename T>
class DropoutGradKernelGPU final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  DropoutGradKernelGPU() = default;
  ~DropoutGradKernelGPU() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float scale = ctx->Attr<float>("scale");
    const int64_t elem_cnt = dy->shape().elem_cnt();
    OF_CUDA_CHECK((cuda::elementwise::Binary(MaskAndScaleFunctor<T>(scale), elem_cnt,
                                             dx->mut_dptr<T>(), dy->dptr<T>(), mask->dptr<int8_t>(),
                                             ctx->device_ctx()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_GRAD_KERNEL_GPU(dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                          \
      .SetCreateFn<DropoutGradKernelGPU<dtype>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_GRAD_KERNEL_GPU(half)
REGISTER_DROPOUT_GRAD_KERNEL_GPU(float)
REGISTER_DROPOUT_GRAD_KERNEL_GPU(double)

}  // namespace

}  // namespace oneflow
