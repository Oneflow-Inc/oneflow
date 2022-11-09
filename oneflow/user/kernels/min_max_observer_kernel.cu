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
#include "oneflow/core/ep/cuda/cuda_stream.h"

#include <float.h>

namespace oneflow {

namespace {

// NOTE(Liang Depeng): refer to
// https://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda
template<typename T>
__global__ void ReduceMaxMinPerLayer(const T* input_ptr, const int64_t elements, T* max_ptr,
                                     T* min_ptr) {
  extern __shared__ unsigned char shared_max_min_memory[];
  T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
  T* shared_min = shared_max + blockDim.x;

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
__global__ void ReduceMaxMinPerChannel(const T* input_ptr, const int64_t elements,
                                       const int64_t num_channels, const int64_t panel_size,
                                       T* max_ptr, T* min_ptr) {
  extern __shared__ unsigned char shared_max_min_memory[];
  T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
  T* shared_min = shared_max + blockDim.x;

  int64_t cur_channel = blockIdx.x;
  int64_t tid = threadIdx.x;

  while (cur_channel < num_channels) {
    shared_max[tid] = -FLT_MAX;
    shared_min[tid] = -FLT_MAX;

    int64_t index = (panel_size * cur_channel) + tid;
    int64_t end = panel_size * (cur_channel + 1);

    while (index < end && index < elements) {
      shared_max[tid] = max(shared_max[tid], input_ptr[index]);
      shared_min[tid] = max(shared_min[tid], -input_ptr[index]);
      index += blockDim.x;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        shared_min[tid] = max(shared_min[tid], shared_min[tid + s]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      cuda::atomic::Max(&max_ptr[cur_channel], shared_max[0]);
      cuda::atomic::Max(&min_ptr[cur_channel], shared_min[0]);
    }

    // __syncthreads();
    cur_channel += gridDim.x;
  }
}

template<typename T>
__global__ void InitMaxMin(const int64_t elements, T* max_ptr, T* min_ptr) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    max_ptr[gid] = -FLT_MAX;
    min_ptr[gid] = -FLT_MAX;
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void CalScaleZeroPointSymmetric(const T* max_ptr, const T* min_ptr,
                                           const int64_t elements, const double quantization_bit,
                                           T* scale, T* zero_point) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    T weight_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    scale[gid] = weight_max / denominator;
    zero_point[gid] = 0;
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void CalScaleZeroPointAffine(const T* max_ptr, const T* min_ptr, const int64_t elements,
                                        const double quantization_bit, T* scale, T* zero_point) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;
    T min = -min_ptr[gid];
    T s = (max_ptr[gid] - min) / denominator;
    scale[gid] = s;
    zero_point[gid] = -nearbyint(min / s);
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void CalScaleZeroPointCambricon(const T* max_ptr, const T* min_ptr,
                                           const int64_t elements, const double quantization_bit,
                                           T* scale, T* zero_point) {
  int64_t tid = threadIdx.x;
  int64_t gid = (blockDim.x * blockIdx.x) + tid;

  while (gid < elements) {
    T weight_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));
    // T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    scale[gid] = floor(log2(weight_max)) - (quantization_bit - 2);
    zero_point[gid] = 0;
    gid += gridDim.x * blockDim.x;
  }
}

ep::CudaLaunchConfig GetLaunchConfig(ep::CudaStream* stream, size_t thread_num,
                                     size_t shared_mem_size) {
  ep::CudaLaunchConfig config;
  stream->InitLaunchConfigWithWaves(&config, thread_num, kCudaThreadsNumPerBlock, 1);
  config.shared_mem_size = shared_mem_size;
  return config;
}

}  // namespace

#define LAUNCH_CUDA_KERNEL(func, stream, thread_num, shared_mem_size, ...) \
  (stream)->LaunchKernel(func, GetLaunchConfig((stream), thread_num, shared_mem_size), __VA_ARGS__);

template<typename T>
class GpuMinMaxObserverKernel final : public user_op::OpKernel {
 public:
  GpuMinMaxObserverKernel() = default;
  ~GpuMinMaxObserverKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor* zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const bool per_layer_quantization = ctx->Attr<bool>("per_layer_quantization");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    const int64_t elements = in->shape_view().elem_cnt();
    const int64_t channel = scale->shape_view().At(0);
    const int64_t panel_size = elements / channel;
    T* max_ptr = tmp_buffer->mut_dptr<T>();
    T* min_ptr = max_ptr + channel;
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    LAUNCH_CUDA_KERNEL((InitMaxMin<T>), cuda_stream, channel, 0, channel, max_ptr, min_ptr);

    if (per_layer_quantization) {
      LAUNCH_CUDA_KERNEL((ReduceMaxMinPerLayer<T>), cuda_stream, elements,
                         kCudaThreadsNumPerBlock * 2 * sizeof(T), in->dptr<T>(), elements, max_ptr,
                         min_ptr);
    } else {  // per-channel quantization
      // NOTE(Liang Depeng): each block of threads will be responsible for
      //                     computing the max and min values of the whole channel.
      LAUNCH_CUDA_KERNEL((ReduceMaxMinPerChannel<T>), cuda_stream,
                         channel * kCudaThreadsNumPerBlock, kCudaThreadsNumPerBlock * 2 * sizeof(T),
                         in->dptr<T>(), elements, channel, panel_size, max_ptr, min_ptr);
    }

    if (quantization_formula == "google") {
      if (quantization_scheme == "symmetric") {
        LAUNCH_CUDA_KERNEL((CalScaleZeroPointSymmetric<T>), cuda_stream, channel, 0, max_ptr,
                           min_ptr, channel, static_cast<double>(quantization_bit),
                           scale->mut_dptr<T>(), zero_point->mut_dptr<T>());
      } else {  // quantization_scheme == "affine"
        LAUNCH_CUDA_KERNEL((CalScaleZeroPointAffine<T>), cuda_stream, channel, 0, max_ptr, min_ptr,
                           channel, static_cast<double>(quantization_bit), scale->mut_dptr<T>(),
                           zero_point->mut_dptr<T>());
      }
    } else if (quantization_formula == "cambricon") {
      if (!per_layer_quantization) {
        UNIMPLEMENTED() << " per-channel mode is not supported in cambricon scheme";
      }
      LAUNCH_CUDA_KERNEL((CalScaleZeroPointCambricon<T>), cuda_stream, channel, 0, max_ptr, min_ptr,
                         channel, static_cast<double>(quantization_bit), scale->mut_dptr<T>(),
                         zero_point->mut_dptr<T>());
    } else {
      UNIMPLEMENTED();
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MIN_MAX_OBSERVER_KERNEL(dtype)                                         \
  REGISTER_USER_KERNEL("min_max_observer")                                              \
      .SetCreateFn<GpuMinMaxObserverKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                     \
        size_t tmp_buffer_size = 1;                                                     \
        if (ctx->Attr<bool>("per_layer_quantization") == false) {                       \
          const Shape& in_shape = ctx->InputShape("in", 0);                             \
          tmp_buffer_size = in_shape.At(0);                                             \
        }                                                                               \
        return 2 * tmp_buffer_size * sizeof(dtype);                                     \
      })

REGISTER_MIN_MAX_OBSERVER_KERNEL(float);
REGISTER_MIN_MAX_OBSERVER_KERNEL(double);

}  // namespace oneflow
