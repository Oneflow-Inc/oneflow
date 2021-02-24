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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/cuda/atomic.cuh"

#include <float.h>

namespace oneflow {

namespace {

// NOTE(Liang Depeng): refer to
// https://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda
template<typename T>
__global__ void ReduceMaxMinPerLayer(const T *input_ptr, const int64_t elements, T *max_ptr,
                                     T *min_ptr) {
  extern __shared__ unsigned char shared_max_min_memory[];
  T *shared_max = reinterpret_cast<T *>(shared_max_min_memory);
  T *shared_min = shared_max + blockDim.x;

  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;
  shared_max[tid] = -FLT_MAX;
  shared_min[tid] = -FLT_MAX;

  while (gid < elements) {
    shared_max[tid] = max(shared_max[tid], input_ptr[gid]);
    shared_min[tid] = max(shared_min[tid], -input_ptr[gid]);
    gid += gridDim.x * blockDim.x;
  }
  __syncthreads();
  gid = (blockDim.x * blockIdx.x) + tid;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && gid < elements) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
      shared_min[tid] = max(shared_min[tid], shared_min[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    cuda::atomic::Max(max_ptr, shared_max[0]);
    cuda::atomic::Max(min_ptr, shared_min[0]);
  }
}

template<typename T>
__global__ void InitMaxMin(const int64_t elements, T *max_ptr, T *min_ptr) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    max_ptr[gid] = -FLT_MAX;
    min_ptr[gid] = -FLT_MAX;
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void CalScaleZeroPointSymmetric(const int64_t elements, const double quantization_bit,
                                           const float momentum, const T *max_ptr, const T *min_ptr,
                                           T *moving_max_ptr, T *moving_min_ptr, T *scale,
                                           T *zero_point) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    T activation_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;

    if (moving_max_ptr[gid] == 0)
      moving_max_ptr[gid] = activation_max;
    else
      moving_max_ptr[gid] = moving_max_ptr[gid] * momentum + activation_max * (1 - momentum);

    // NOTE(Liang Depeng): symmetric quantization only use moving_max to calculate the scale
    moving_min_ptr[gid] = moving_max_ptr[gid];

    scale[gid] = moving_max_ptr[gid] / denominator;
    zero_point[gid] = 0;
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void CalFreezeScaleZeroPointSymmetric(const int64_t elements,
                                                 const double quantization_bit,
                                                 const float momentum, const T *moving_max_ptr,
                                                 T *scale, T *zero_point) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    scale[gid] = moving_max_ptr[gid] / denominator;
    zero_point[gid] = 0;
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void CalScaleZeroPointAffine(const int64_t elements, const double quantization_bit,
                                        const float momentum, const T *max_ptr, const T *min_ptr,
                                        T *moving_max_ptr, T *moving_min_ptr, T *scale,
                                        T *zero_point) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;

    if (moving_max_ptr[gid] == 0)
      moving_max_ptr[gid] = max_ptr[gid];
    else
      moving_max_ptr[gid] = moving_max_ptr[gid] * momentum + max_ptr[gid] * (1 - momentum);

    if (moving_min_ptr[gid] == 0)
      moving_min_ptr[gid] = -min_ptr[gid];
    else
      moving_min_ptr[gid] = moving_min_ptr[gid] * momentum + -min_ptr[gid] * (1 - momentum);

    T min = moving_min_ptr[gid];
    T s = (moving_max_ptr[gid] - min) / denominator;

    scale[gid] = s;
    zero_point[gid] = -min / s;
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void CalFreezeScaleZeroPointAffine(const int64_t elements, const double quantization_bit,
                                              const float momentum, const T *moving_max_ptr,
                                              const T *moving_min_ptr, T *scale, T *zero_point) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;

    T min = moving_min_ptr[gid];
    T s = (moving_max_ptr[gid] - min) / denominator;

    scale[gid] = s;
    zero_point[gid] = -min / s;
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void CalScaleZeroPointCambricon(const int64_t elements, const double quantization_bit,
                                           const float momentum, const T *max_ptr, const T *min_ptr,
                                           T *moving_max_ptr, T *moving_min_ptr, T *scale,
                                           T *zero_point) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    T activation_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));

    if (moving_max_ptr[gid] == 0)
      moving_max_ptr[gid] = activation_max;
    else
      moving_max_ptr[gid] = moving_max_ptr[gid] * momentum + activation_max * (1 - momentum);

    // NOTE(Liang Depeng): cambricon quantization only use moving_max to calculate the scale
    moving_min_ptr[gid] = moving_max_ptr[gid];

    scale[gid] = floor(log2(moving_max_ptr[gid])) - (quantization_bit - 2);
    zero_point[gid] = 0;
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void CalFreezeScaleZeroPointCambricon(const int64_t elements,
                                                 const double quantization_bit,
                                                 const float momentum, const T *moving_max_ptr,
                                                 T *scale, T *zero_point) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    scale[gid] = floor(log2(moving_max_ptr[gid])) - (quantization_bit - 2);
    zero_point[gid] = 0;
    gid += gridDim.x * blockDim.x;
  }
}

}  // namespace

#define LAUNCH_CUDA_KERNEL(func, device_ctx_ptr, thread_num, shared_mem_size, ...)     \
  func<<<SMBlocksNum4ThreadsNum(thread_num), kCudaThreadsNumPerBlock, shared_mem_size, \
         (device_ctx_ptr)->cuda_stream()>>>(__VA_ARGS__)

template<typename T>
class GpuMovingAverageMinMaxObserverKernel final : public user_op::OpKernel {
 public:
  GpuMovingAverageMinMaxObserverKernel() = default;
  ~GpuMovingAverageMinMaxObserverKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor *current_train_step =
        ctx->Tensor4ArgNameAndIndex("current_train_step", 0);
    user_op::Tensor *moving_max = ctx->Tensor4ArgNameAndIndex("moving_max", 0);
    user_op::Tensor *moving_min = ctx->Tensor4ArgNameAndIndex("moving_min", 0);
    user_op::Tensor *scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor *zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);
    user_op::Tensor *tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const bool is_training = ctx->Attr<bool>("training");
    const int64_t stop_update_after_iters = ctx->Attr<int64_t>("stop_update_after_iters");
    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const float momentum = ctx->Attr<float>("momentum");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    int64_t elements = in->shape().elem_cnt();
    T *max_ptr = tmp_buffer->mut_dptr<T>();
    T *min_ptr = max_ptr + 1;

    int64_t *host_current_train_step_ptr = new int64_t[current_train_step->shape().elem_cnt()];
    OF_CUDA_CHECK(cudaMemcpy(host_current_train_step_ptr, current_train_step->dptr<int64_t>(),
                             current_train_step->shape().elem_cnt() * sizeof(int64_t),
                             cudaMemcpyDefault));

    if (*host_current_train_step_ptr <= stop_update_after_iters && is_training) {
      LAUNCH_CUDA_KERNEL((InitMaxMin<T>), ctx->device_ctx(), 1, 0, 1, max_ptr, min_ptr);
      LAUNCH_CUDA_KERNEL((ReduceMaxMinPerLayer<T>), ctx->device_ctx(), elements,
                         kCudaThreadsNumPerBlock * 2 * sizeof(T), in->dptr<T>(), elements, max_ptr,
                         min_ptr);
    }

    if (quantization_formula == "google") {
      if (quantization_scheme == "symmetric") {
        if (*host_current_train_step_ptr <= stop_update_after_iters) {
          LAUNCH_CUDA_KERNEL((CalScaleZeroPointSymmetric<T>), ctx->device_ctx(), 1, 0, 1,
                             static_cast<double>(quantization_bit), momentum, max_ptr, min_ptr,
                             moving_max->mut_dptr<T>(), moving_min->mut_dptr<T>(),
                             scale->mut_dptr<T>(), zero_point->mut_dptr<T>());
        } else {
          LAUNCH_CUDA_KERNEL((CalFreezeScaleZeroPointSymmetric<T>), ctx->device_ctx(), 1, 0, 1,
                             static_cast<double>(quantization_bit), momentum, moving_max->dptr<T>(),
                             scale->mut_dptr<T>(), zero_point->mut_dptr<T>());
        }
      } else {  // quantization_scheme == "affine"
        if (*host_current_train_step_ptr <= stop_update_after_iters) {
          LAUNCH_CUDA_KERNEL((CalScaleZeroPointAffine<T>), ctx->device_ctx(), 1, 0, 1,
                             static_cast<double>(quantization_bit), momentum, max_ptr, min_ptr,
                             moving_max->mut_dptr<T>(), moving_min->mut_dptr<T>(),
                             scale->mut_dptr<T>(), zero_point->mut_dptr<T>());
        } else {
          LAUNCH_CUDA_KERNEL((CalFreezeScaleZeroPointAffine<T>), ctx->device_ctx(), 1, 0, 1,
                             static_cast<double>(quantization_bit), momentum, moving_max->dptr<T>(),
                             moving_min->dptr<T>(), scale->mut_dptr<T>(),
                             zero_point->mut_dptr<T>());
        }
      }
    } else if (quantization_formula == "cambricon") {
      if (*host_current_train_step_ptr <= stop_update_after_iters) {
        LAUNCH_CUDA_KERNEL((CalScaleZeroPointCambricon<T>), ctx->device_ctx(), 1, 0, 1,
                           static_cast<double>(quantization_bit), momentum, max_ptr, min_ptr,
                           moving_max->mut_dptr<T>(), moving_min->mut_dptr<T>(),
                           scale->mut_dptr<T>(), zero_point->mut_dptr<T>());
      } else {
        LAUNCH_CUDA_KERNEL((CalFreezeScaleZeroPointCambricon<T>), ctx->device_ctx(), 1, 0, 1,
                           static_cast<double>(quantization_bit), momentum, moving_max->dptr<T>(),
                           scale->mut_dptr<T>(), zero_point->mut_dptr<T>());
      }
    } else {
      UNIMPLEMENTED();
    }

    delete[] host_current_train_step_ptr;
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MOVING_AVERAGE_MIN_MAX_OBSERVER_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("moving_average_min_max_observer")                              \
      .SetCreateFn<GpuMovingAverageMinMaxObserverKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                   \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext *ctx) -> size_t { return 2 * sizeof(dtype); })

REGISTER_MOVING_AVERAGE_MIN_MAX_OBSERVER_KERNEL(float);
REGISTER_MOVING_AVERAGE_MIN_MAX_OBSERVER_KERNEL(double);

}  // namespace oneflow
