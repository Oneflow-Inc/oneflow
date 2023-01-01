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
#include "oneflow/user/kernels/model_update_kernel_util.h"
#include "oneflow/user/kernels/multi_tensor_model_update_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

constexpr int kBlockSize = 256;
constexpr int kUnrollSize = 4;

unsigned int ComputeGridSize(ep::Stream* stream, const int32_t block_size, const int64_t elem_cnt) {
  auto* cuda_stream = stream->As<ep::CudaStream>();
  const int32_t max_threads_multi_process =
      cuda_stream->device_properties().maxThreadsPerMultiProcessor;
  const int32_t multi_processor_count = cuda_stream->device_properties().multiProcessorCount;
  unsigned int blocks_per_sm = max_threads_multi_process / block_size;
  unsigned int grid_size = ((elem_cnt + block_size - 1) / block_size);
  grid_size = std::min((unsigned int)multi_processor_count * blocks_per_sm, grid_size);
  return grid_size;
}

template<typename T, typename G, int N>
__global__ void MultiTensorSGDUpdateGpu(int64_t num_tensor, T scale, const float l1, const float l2,
                                        const float weight_decay, float learning_rate_val,
                                        float lr_scale, const float* learning_rate,
                                        const T* scale_by_ptr, const int64_t* skip_if,
                                        TensorTupleParams<N> tensor_tuple_params) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  learning_rate_val *= lr_scale;
  int64_t v_block_id = blockIdx.x;
  for (int64_t tensor_idx = 0; tensor_idx < num_tensor; tensor_idx++) {
    const int64_t tensor_elem_cnt = tensor_tuple_params.sizes[tensor_idx];
    T* model_ptr = (T*)tensor_tuple_params.ptr[0][tensor_idx];
    G* model_diff_ptr = (G*)tensor_tuple_params.ptr[1][tensor_idx];
    half* model_copy_ptr = nullptr;
    if (N == 3) { model_copy_ptr = (half*)tensor_tuple_params.ptr[2][tensor_idx]; }

    for (int64_t i = v_block_id * blockDim.x * kUnrollSize + threadIdx.x; i < tensor_elem_cnt;
         i += blockDim.x * gridDim.x * kUnrollSize) {
      T model_val[kUnrollSize] = {0};
      G model_diff[kUnrollSize] = {0};

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          model_val[ilp] = *(model_ptr + actual_idx);
          model_diff[ilp] = *(model_diff_ptr + actual_idx);
        }
      }

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          T model_diff_t = CastScaleRegularizeGradientFunctor<T, G>()(
              model_diff[ilp], model_val[ilp], scale, l1, l2);
          model_val[ilp] =
              model_val[ilp] - learning_rate_val * (model_diff_t + weight_decay * model_val[ilp]);
        }
      }

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          *(model_ptr + actual_idx) = model_val[ilp];
          if (N == 3) { *(model_copy_ptr + actual_idx) = static_cast<half>(model_val[ilp]); }
        }
      }
    }
    v_block_id -= tensor_tuple_params.block_offset[tensor_idx];
    if (v_block_id < 0) { v_block_id += gridDim.x; }
  }
}

template<typename T, typename G>
struct MultiTensorSGDUpdateKernelUtil<DeviceType::kCUDA, T, G> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, TensorTupleParams<2> tensor_tuple_params);
};

template<typename T, typename G>
void MultiTensorSGDUpdateKernelUtil<DeviceType::kCUDA, T, G>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float weight_decay, float learning_rate_val, float lr_scale, const float* learning_rate,
    const T* scale_by_ptr, const int64_t* skip_if, TensorTupleParams<2> tensor_tuple_params) {
  const unsigned int grid_size =
      ComputeGridSize(stream->As<ep::CudaStream>(), kBlockSize, elem_cnt);
  for (int i = 0; i < n_tensor; i++) {
    tensor_tuple_params.block_offset[i] =
        ((tensor_tuple_params.sizes[i] + kBlockSize * kUnrollSize - 1) / (kBlockSize * kUnrollSize))
        % grid_size;
  }
  MultiTensorSGDUpdateGpu<T, G, 2>
      <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          n_tensor, static_cast<T>(scale), l1, l2, weight_decay, learning_rate_val, lr_scale,
          learning_rate, scale_by_ptr, skip_if, tensor_tuple_params);
}

template<typename T>
struct MultiTensorSGDUpdateKernelUtil<DeviceType::kCUDA, T, float16> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, TensorTupleParams<2> tensor_tuple_params);
};

template<typename T>
void MultiTensorSGDUpdateKernelUtil<DeviceType::kCUDA, T, float16>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float weight_decay, float learning_rate_val, float lr_scale, const float* learning_rate,
    const T* scale_by_ptr, const int64_t* skip_if, TensorTupleParams<2> tensor_tuple_params) {
  MultiTensorSGDUpdateKernelUtil<DeviceType::kCUDA, T, half>::Update(
      stream, elem_cnt, n_tensor, scale, l1, l2, weight_decay, learning_rate_val, lr_scale,
      learning_rate, scale_by_ptr, skip_if, tensor_tuple_params);
}

template struct MultiTensorSGDUpdateKernelUtil<DeviceType::kCUDA, double, double>;
template struct MultiTensorSGDUpdateKernelUtil<DeviceType::kCUDA, float, float>;
template struct MultiTensorSGDUpdateKernelUtil<DeviceType::kCUDA, float, float16>;

template<typename T, typename G, int N>
__global__ void MultiTensorMomentumUpdateGpu(
    int64_t num_tensor, T scale, const float l1, const float l2, const float weight_decay,
    float learning_rate_val, float lr_scale, const float* learning_rate, const T* scale_by_ptr,
    const int64_t* skip_if, const float momentum, const float dampening, const bool nesterov,
    const bool maximize, TensorTupleParams<N> tensor_tuple_params) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  learning_rate_val *= lr_scale;
  int64_t v_block_id = blockIdx.x;
  for (int64_t tensor_idx = 0; tensor_idx < num_tensor; tensor_idx++) {
    const int64_t tensor_elem_cnt = tensor_tuple_params.sizes[tensor_idx];
    T* model_ptr = (T*)tensor_tuple_params.ptr[0][tensor_idx];
    G* model_diff_ptr = (G*)tensor_tuple_params.ptr[1][tensor_idx];
    T* momentum_buf_ptr = (T*)tensor_tuple_params.ptr[2][tensor_idx];
    half* model_copy_ptr = nullptr;
    if (N == 4) { model_copy_ptr = (half*)tensor_tuple_params.ptr[3][tensor_idx]; }

    for (int64_t i = v_block_id * blockDim.x * kUnrollSize + threadIdx.x; i < tensor_elem_cnt;
         i += blockDim.x * gridDim.x * kUnrollSize) {
      T model_val[kUnrollSize] = {0};
      G model_diff[kUnrollSize] = {0};
      T momentum_buf[kUnrollSize] = {0};

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          model_val[ilp] = *(model_ptr + actual_idx);
          model_diff[ilp] = *(model_diff_ptr + actual_idx);
          momentum_buf[ilp] = *(momentum_buf_ptr + actual_idx);
        }
      }

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          T model_diff_t = CastScaleRegularizeGradientFunctor<T, G>()(
              model_diff[ilp], model_val[ilp], scale, l1, l2);

          if (weight_decay != 0.f) { model_diff_t += weight_decay * model_val[ilp]; }

          momentum_buf[ilp] = momentum * momentum_buf[ilp] + (1.f - dampening) * model_diff_t;

          if (nesterov)
            model_diff_t += momentum * momentum_buf[ilp];
          else
            model_diff_t = momentum_buf[ilp];

          T alpha = -learning_rate_val;
          if (maximize) alpha = learning_rate_val;
          model_val[ilp] += alpha * model_diff_t;
        }
      }

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          *(model_ptr + actual_idx) = model_val[ilp];
          *(momentum_buf_ptr + actual_idx) = momentum_buf[ilp];
          if (N == 4) { *(model_copy_ptr + actual_idx) = static_cast<half>(model_val[ilp]); }
        }
      }
    }
    v_block_id -= tensor_tuple_params.block_offset[tensor_idx];
    if (v_block_id < 0) { v_block_id += gridDim.x; }
  }
}

template<typename T, typename G>
struct MultiTensorMomentumUpdateKernelUtil<DeviceType::kCUDA, T, G> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const float momentum, const float dampening,
                     const bool nesterov, const bool maximize,
                     TensorTupleParams<3> tensor_tuple_params);
};

template<typename T, typename G>
void MultiTensorMomentumUpdateKernelUtil<DeviceType::kCUDA, T, G>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float weight_decay, float learning_rate_val, float lr_scale, const float* learning_rate,
    const T* scale_by_ptr, const int64_t* skip_if, const float momentum, const float dampening,
    const bool nesterov, const bool maximize, TensorTupleParams<3> tensor_tuple_params) {
  const unsigned int grid_size =
      ComputeGridSize(stream->As<ep::CudaStream>(), kBlockSize, elem_cnt);
  for (int i = 0; i < n_tensor; i++) {
    tensor_tuple_params.block_offset[i] =
        ((tensor_tuple_params.sizes[i] + kBlockSize * kUnrollSize - 1) / (kBlockSize * kUnrollSize))
        % grid_size;
  }
  MultiTensorMomentumUpdateGpu<T, G, 3>
      <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          n_tensor, static_cast<T>(scale), l1, l2, weight_decay, learning_rate_val, lr_scale,
          learning_rate, scale_by_ptr, skip_if, momentum, dampening, nesterov, maximize,
          tensor_tuple_params);
}

template<typename T>
struct MultiTensorMomentumUpdateKernelUtil<DeviceType::kCUDA, T, float16> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const float momentum, const float dampening,
                     const bool nesterov, const bool maximize,
                     TensorTupleParams<3> tensor_tuple_params);
};

template<typename T>
void MultiTensorMomentumUpdateKernelUtil<DeviceType::kCUDA, T, float16>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float weight_decay, float learning_rate_val, float lr_scale, const float* learning_rate,
    const T* scale_by_ptr, const int64_t* skip_if, const float momentum, const float dampening,
    const bool nesterov, const bool maximize, TensorTupleParams<3> tensor_tuple_params) {
  MultiTensorMomentumUpdateKernelUtil<DeviceType::kCUDA, T, half>::Update(
      stream, elem_cnt, n_tensor, scale, l1, l2, weight_decay, learning_rate_val, lr_scale,
      learning_rate, scale_by_ptr, skip_if, momentum, dampening, nesterov, maximize,
      tensor_tuple_params);
}

template struct MultiTensorMomentumUpdateKernelUtil<DeviceType::kCUDA, double, double>;
template struct MultiTensorMomentumUpdateKernelUtil<DeviceType::kCUDA, float, float>;
template struct MultiTensorMomentumUpdateKernelUtil<DeviceType::kCUDA, float, float16>;

template<typename T, typename G, int N>
__global__ void MultiTensorAdamUpdateGpu(int64_t num_tensor, T scale, float l1, float l2,
                                         float beta1, float beta2, float epsilon,
                                         float weight_decay, bool amsgrad, bool do_bias_correction,
                                         float learning_rate_val, float bias_correction1_val,
                                         float bias_correction2_val, float lr_scale,
                                         const float* learning_rate, const T* scale_by_ptr,
                                         const int64_t* skip_if, const float* bias_correction1_ptr,
                                         const float* bias_correction2_ptr,
                                         TensorTupleParams<N> tensor_tuple_params) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  if (bias_correction1_ptr != nullptr) { bias_correction1_val = *bias_correction1_ptr; }
  if (bias_correction2_ptr != nullptr) { bias_correction2_val = *bias_correction2_ptr; }

  learning_rate_val *= lr_scale;
  int64_t v_block_id = blockIdx.x;
  for (int64_t tensor_idx = 0; tensor_idx < num_tensor; tensor_idx++) {
    const int64_t tensor_elem_cnt = tensor_tuple_params.sizes[tensor_idx];
    T* model_ptr = (T*)tensor_tuple_params.ptr[0][tensor_idx];
    G* model_diff_ptr = (G*)tensor_tuple_params.ptr[1][tensor_idx];
    T* m_ptr = (T*)tensor_tuple_params.ptr[2][tensor_idx];
    T* v_ptr = (T*)tensor_tuple_params.ptr[3][tensor_idx];
    half* model_copy_ptr = nullptr;
    if (N == 5) { model_copy_ptr = (half*)tensor_tuple_params.ptr[4][tensor_idx]; }

    for (int64_t i = v_block_id * blockDim.x * kUnrollSize + threadIdx.x; i < tensor_elem_cnt;
         i += blockDim.x * gridDim.x * kUnrollSize) {
      T model_val[kUnrollSize] = {0};
      T m_val[kUnrollSize] = {0};
      T v_val[kUnrollSize] = {0};
      G model_diff[kUnrollSize] = {0};

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          model_val[ilp] = *(model_ptr + actual_idx);
          m_val[ilp] = *(m_ptr + actual_idx);
          v_val[ilp] = *(v_ptr + actual_idx);
          model_diff[ilp] = *(model_diff_ptr + actual_idx);
        }
      }

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          T model_diff_t = CastScaleRegularizeGradientFunctor<T, G>()(
              model_diff[ilp], model_val[ilp], scale, l1, l2);

          m_val[ilp] = beta1 * m_val[ilp] + (1 - beta1) * model_diff_t;
          v_val[ilp] = beta2 * v_val[ilp] + (1 - beta2) * model_diff_t * model_diff_t;

          T denom = (sqrt(v_val[ilp]) / sqrt(bias_correction2_val)) + epsilon;
          const T step_size = learning_rate_val / bias_correction1_val;
          model_val[ilp] = model_val[ilp] - step_size * (m_val[ilp] / denom)
                           - learning_rate_val * weight_decay * model_val[ilp];
        }
      }

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          *(model_ptr + actual_idx) = model_val[ilp];
          *(m_ptr + actual_idx) = m_val[ilp];
          *(v_ptr + actual_idx) = v_val[ilp];
          if (N == 5) { *(model_copy_ptr + actual_idx) = static_cast<half>(model_val[ilp]); }
        }
      }
    }
    v_block_id -= tensor_tuple_params.block_offset[tensor_idx];
    if (v_block_id < 0) { v_block_id += gridDim.x; }
  }
}

template<typename T, typename G>
struct MultiTensorAdamUpdateKernelUtil<DeviceType::kCUDA, T, G> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float beta1, float beta2, float epsilon,
                     float weight_decay, bool amsgrad, bool do_bias_correction,
                     float learning_rate_val, float bias_correction1_val,
                     float bias_correction2_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const float* bias_correction1,
                     const float* bias_correction2, TensorTupleParams<4> tensor_tuple_params);
};

template<typename T, typename G>
void MultiTensorAdamUpdateKernelUtil<DeviceType::kCUDA, T, G>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float beta1, float beta2, float epsilon, float weight_decay, bool amsgrad,
    bool do_bias_correction, float learning_rate_val, float bias_correction1_val,
    float bias_correction2_val, float lr_scale, const float* learning_rate, const T* scale_by_ptr,
    const int64_t* skip_if, const float* bias_correction1, const float* bias_correction2,
    TensorTupleParams<4> tensor_tuple_params) {
  const unsigned int grid_size =
      ComputeGridSize(stream->As<ep::CudaStream>(), kBlockSize, elem_cnt);
  for (int i = 0; i < n_tensor; i++) {
    tensor_tuple_params.block_offset[i] =
        ((tensor_tuple_params.sizes[i] + kBlockSize * kUnrollSize - 1) / (kBlockSize * kUnrollSize))
        % grid_size;
  }
  MultiTensorAdamUpdateGpu<T, G>
      <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          n_tensor, scale, l1, l2, beta1, beta2, epsilon, weight_decay, amsgrad, do_bias_correction,
          learning_rate_val, bias_correction1_val, bias_correction2_val, lr_scale, learning_rate,
          scale_by_ptr, skip_if, bias_correction1, bias_correction2, tensor_tuple_params);
}

template<typename T>
struct MultiTensorAdamUpdateKernelUtil<DeviceType::kCUDA, T, float16> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float beta1, float beta2, float epsilon,
                     float weight_decay, bool amsgrad, bool do_bias_correction,
                     float learning_rate_val, float bias_correction1_val,
                     float bias_correction2_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const float* bias_correction1,
                     const float* bias_correction2, TensorTupleParams<4> tensor_tuple_params);
};

template<typename T>
void MultiTensorAdamUpdateKernelUtil<DeviceType::kCUDA, T, float16>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float beta1, float beta2, float epsilon, float weight_decay, bool amsgrad,
    bool do_bias_correction, float learning_rate_val, float bias_correction1_val,
    float bias_correction2_val, float lr_scale, const float* learning_rate, const T* scale_by_ptr,
    const int64_t* skip_if, const float* bias_correction1, const float* bias_correction2,
    TensorTupleParams<4> tensor_tuple_params) {
  MultiTensorAdamUpdateKernelUtil<DeviceType::kCUDA, T, half>::Update(
      stream, elem_cnt, n_tensor, scale, l1, l2, beta1, beta2, epsilon, weight_decay, amsgrad,
      do_bias_correction, learning_rate_val, bias_correction1_val, bias_correction2_val, lr_scale,
      learning_rate, scale_by_ptr, skip_if, bias_correction1, bias_correction2,
      tensor_tuple_params);
}

template struct MultiTensorAdamUpdateKernelUtil<DeviceType::kCUDA, double, double>;
template struct MultiTensorAdamUpdateKernelUtil<DeviceType::kCUDA, float, float>;
template struct MultiTensorAdamUpdateKernelUtil<DeviceType::kCUDA, float, float16>;

template<typename T, typename G>
struct MultiTensorSGDUpdateWithCastKernelUtil<DeviceType::kCUDA, T, G> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, TensorTupleParams<3> tensor_tuple_params);
};

template<typename T, typename G>
void MultiTensorSGDUpdateWithCastKernelUtil<DeviceType::kCUDA, T, G>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float weight_decay, float learning_rate_val, float lr_scale, const float* learning_rate,
    const T* scale_by_ptr, const int64_t* skip_if, TensorTupleParams<3> tensor_tuple_params) {
  const unsigned int grid_size =
      ComputeGridSize(stream->As<ep::CudaStream>(), kBlockSize, elem_cnt);
  for (int i = 0; i < n_tensor; i++) {
    tensor_tuple_params.block_offset[i] =
        ((tensor_tuple_params.sizes[i] + kBlockSize * kUnrollSize - 1) / (kBlockSize * kUnrollSize))
        % grid_size;
  }
  MultiTensorSGDUpdateGpu<T, G, 3>
      <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          n_tensor, static_cast<T>(scale), l1, l2, weight_decay, learning_rate_val, lr_scale,
          learning_rate, scale_by_ptr, skip_if, tensor_tuple_params);
}

template<typename T>
struct MultiTensorSGDUpdateWithCastKernelUtil<DeviceType::kCUDA, T, float16> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, TensorTupleParams<3> tensor_tuple_params);
};

template<typename T>
void MultiTensorSGDUpdateWithCastKernelUtil<DeviceType::kCUDA, T, float16>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float weight_decay, float learning_rate_val, float lr_scale, const float* learning_rate,
    const T* scale_by_ptr, const int64_t* skip_if, TensorTupleParams<3> tensor_tuple_params) {
  MultiTensorSGDUpdateWithCastKernelUtil<DeviceType::kCUDA, T, half>::Update(
      stream, elem_cnt, n_tensor, scale, l1, l2, weight_decay, learning_rate_val, lr_scale,
      learning_rate, scale_by_ptr, skip_if, tensor_tuple_params);
}

template struct MultiTensorSGDUpdateWithCastKernelUtil<DeviceType::kCUDA, float, float>;
template struct MultiTensorSGDUpdateWithCastKernelUtil<DeviceType::kCUDA, float, float16>;

template<typename T, typename G>
struct MultiTensorMomentumUpdateWithCastKernelUtil<DeviceType::kCUDA, T, G> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const float momentum, const float dampening,
                     const bool nesterov, const bool maximize,
                     TensorTupleParams<4> tensor_tuple_params);
};

template<typename T, typename G>
void MultiTensorMomentumUpdateWithCastKernelUtil<DeviceType::kCUDA, T, G>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float weight_decay, float learning_rate_val, float lr_scale, const float* learning_rate,
    const T* scale_by_ptr, const int64_t* skip_if, const float momentum, const float dampening,
    const bool nesterov, const bool maximize, TensorTupleParams<4> tensor_tuple_params) {
  const unsigned int grid_size =
      ComputeGridSize(stream->As<ep::CudaStream>(), kBlockSize, elem_cnt);
  for (int i = 0; i < n_tensor; i++) {
    tensor_tuple_params.block_offset[i] =
        ((tensor_tuple_params.sizes[i] + kBlockSize * kUnrollSize - 1) / (kBlockSize * kUnrollSize))
        % grid_size;
  }
  MultiTensorMomentumUpdateGpu<T, G, 4>
      <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          n_tensor, static_cast<T>(scale), l1, l2, weight_decay, learning_rate_val, lr_scale,
          learning_rate, scale_by_ptr, skip_if, momentum, dampening, nesterov, maximize,
          tensor_tuple_params);
}

template<typename T>
struct MultiTensorMomentumUpdateWithCastKernelUtil<DeviceType::kCUDA, T, float16> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const float momentum, const float dampening,
                     const bool nesterov, const bool maximize,
                     TensorTupleParams<4> tensor_tuple_params);
};

template<typename T>
void MultiTensorMomentumUpdateWithCastKernelUtil<DeviceType::kCUDA, T, float16>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float weight_decay, float learning_rate_val, float lr_scale, const float* learning_rate,
    const T* scale_by_ptr, const int64_t* skip_if, const float momentum, const float dampening,
    const bool nesterov, const bool maximize, TensorTupleParams<4> tensor_tuple_params) {
  MultiTensorMomentumUpdateWithCastKernelUtil<DeviceType::kCUDA, T, half>::Update(
      stream, elem_cnt, n_tensor, scale, l1, l2, weight_decay, learning_rate_val, lr_scale,
      learning_rate, scale_by_ptr, skip_if, momentum, dampening, nesterov, maximize,
      tensor_tuple_params);
}

template struct MultiTensorMomentumUpdateWithCastKernelUtil<DeviceType::kCUDA, float, float>;
template struct MultiTensorMomentumUpdateWithCastKernelUtil<DeviceType::kCUDA, float, float16>;

template<typename T, typename G>
struct MultiTensorAdamUpdateWithCastKernelUtil<DeviceType::kCUDA, T, G> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float beta1, float beta2, float epsilon,
                     float weight_decay, bool amsgrad, bool do_bias_correction,
                     float learning_rate_val, float bias_correction1_val,
                     float bias_correction2_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const float* bias_correction1,
                     const float* bias_correction2, TensorTupleParams<5> tensor_tuple_params);
};

template<typename T, typename G>
void MultiTensorAdamUpdateWithCastKernelUtil<DeviceType::kCUDA, T, G>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float beta1, float beta2, float epsilon, float weight_decay, bool amsgrad,
    bool do_bias_correction, float learning_rate_val, float bias_correction1_val,
    float bias_correction2_val, float lr_scale, const float* learning_rate, const T* scale_by_ptr,
    const int64_t* skip_if, const float* bias_correction1, const float* bias_correction2,
    TensorTupleParams<5> tensor_tuple_params) {
  const unsigned int grid_size =
      ComputeGridSize(stream->As<ep::CudaStream>(), kBlockSize, elem_cnt);
  for (int i = 0; i < n_tensor; i++) {
    tensor_tuple_params.block_offset[i] =
        ((tensor_tuple_params.sizes[i] + kBlockSize * kUnrollSize - 1) / (kBlockSize * kUnrollSize))
        % grid_size;
  }
  MultiTensorAdamUpdateGpu<T, G, 5>
      <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          n_tensor, scale, l1, l2, beta1, beta2, epsilon, weight_decay, amsgrad, do_bias_correction,
          learning_rate_val, bias_correction1_val, bias_correction2_val, lr_scale, learning_rate,
          scale_by_ptr, skip_if, bias_correction1, bias_correction2, tensor_tuple_params);
}

template<typename T>
struct MultiTensorAdamUpdateWithCastKernelUtil<DeviceType::kCUDA, T, float16> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale,
                     float l1, float l2, float beta1, float beta2, float epsilon,
                     float weight_decay, bool amsgrad, bool do_bias_correction,
                     float learning_rate_val, float bias_correction1_val,
                     float bias_correction2_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const float* bias_correction1,
                     const float* bias_correction2, TensorTupleParams<5> tensor_tuple_params);
};

template<typename T>
void MultiTensorAdamUpdateWithCastKernelUtil<DeviceType::kCUDA, T, float16>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, T scale, float l1, float l2,
    float beta1, float beta2, float epsilon, float weight_decay, bool amsgrad,
    bool do_bias_correction, float learning_rate_val, float bias_correction1_val,
    float bias_correction2_val, float lr_scale, const float* learning_rate, const T* scale_by_ptr,
    const int64_t* skip_if, const float* bias_correction1, const float* bias_correction2,
    TensorTupleParams<5> tensor_tuple_params) {
  MultiTensorAdamUpdateWithCastKernelUtil<DeviceType::kCUDA, T, half>::Update(
      stream, elem_cnt, n_tensor, scale, l1, l2, beta1, beta2, epsilon, weight_decay, amsgrad,
      do_bias_correction, learning_rate_val, bias_correction1_val, bias_correction2_val, lr_scale,
      learning_rate, scale_by_ptr, skip_if, bias_correction1, bias_correction2,
      tensor_tuple_params);
}

template struct MultiTensorAdamUpdateWithCastKernelUtil<DeviceType::kCUDA, float, float>;
template struct MultiTensorAdamUpdateWithCastKernelUtil<DeviceType::kCUDA, float, float16>;

template<typename T, int N>
__global__ void MultiTensorYoloModelEmaUpdateGpu(int64_t num_tensor, const float d,
                                                 TensorTupleParams<N> tensor_tuple_params) {
  int64_t v_block_id = blockIdx.x;
  for (int64_t tensor_idx = 0; tensor_idx < num_tensor; tensor_idx++) {
    const int64_t tensor_elem_cnt = tensor_tuple_params.sizes[tensor_idx];
    T* model_ptr = (T*)tensor_tuple_params.ptr[0][tensor_idx];
    T* model_update_ptr = (T*)tensor_tuple_params.ptr[1][tensor_idx];

    for (int64_t i = v_block_id * blockDim.x * kUnrollSize + threadIdx.x; i < tensor_elem_cnt;
         i += blockDim.x * gridDim.x * kUnrollSize) {
      T model_val[kUnrollSize] = {0};
      T model_update_val[kUnrollSize] = {0};

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          model_val[ilp] = *(model_ptr + actual_idx);
          model_update_val[ilp] = *(model_update_ptr + actual_idx);
        }
      }

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          model_val[ilp] *= d;
          model_val[ilp] += (1 - d) * model_update_val[ilp];
        }
      }

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          *(model_ptr + actual_idx) = model_val[ilp];
          *(model_update_ptr + actual_idx) = model_update_val[ilp];
        }
      }
    }
    v_block_id -= tensor_tuple_params.block_offset[tensor_idx];
    if (v_block_id < 0) { v_block_id += gridDim.x; }
  }
}

template<typename T>
struct MultiTensorYoloV5WeightUpdateKernelUtil<DeviceType::kCUDA, T> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, float d,
                     TensorTupleParams<2> tensor_tuple_params);
};

template<>
struct MultiTensorYoloV5WeightUpdateKernelUtil<DeviceType::kCUDA, half> {
  static void Update(ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, float d,
                     TensorTupleParams<2> tensor_tuple_params);
};

template<typename T>
void MultiTensorYoloV5WeightUpdateKernelUtil<DeviceType::kCUDA, T>::Update(
    ep::Stream* stream, const int64_t elem_cnt, const int64_t n_tensor, float d,
    TensorTupleParams<2> tensor_tuple_params) {
  const unsigned int grid_size =
      ComputeGridSize(stream->As<ep::CudaStream>(), kBlockSize, elem_cnt);
  for (int i = 0; i < n_tensor; i++) {
    tensor_tuple_params.block_offset[i] =
        ((tensor_tuple_params.sizes[i] + kBlockSize * kUnrollSize - 1) / (kBlockSize * kUnrollSize))
        % grid_size;
  }
  MultiTensorYoloModelEmaUpdateGpu<T>
      <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          n_tensor, d, tensor_tuple_params);
}

template struct MultiTensorYoloV5WeightUpdateKernelUtil<DeviceType::kCUDA, float>;

}  // namespace oneflow
