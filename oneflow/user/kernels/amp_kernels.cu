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
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__device__ bool IsFinite(T x) {
  return isfinite(x);
}

constexpr int kMaxTuples = 32;

struct TensorTupleParams {
  void* ptr[kMaxTuples];
  int64_t sizes[kMaxTuples];
  int32_t block_offset[kMaxTuples];
};

template<typename T>
__global__ void AmpUpdateScaleImpl(T* current_scale, int* growth_tracker, const T* found_inf,
                                   double growth_factor, double backoff_factor,
                                   int64_t growth_interval) {
  if (*found_inf) {
    *current_scale = (*current_scale) * backoff_factor;
    *growth_tracker = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = (*growth_tracker) + 1;
    if (successful == growth_interval) {
      *current_scale = (*current_scale) * growth_factor;
      *growth_tracker = 0;
    } else {
      *growth_tracker = successful;
    }
  }
}

template<typename T>
__global__ void AmpForEachNonFiniteCheckAndUnscaleImpl(const int n, T* scaled_grad,
                                                       float* found_inf, const float* inv_scale) {
  const auto inv_scale_value = *inv_scale;
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (!IsFinite(scaled_grad[i])) { *found_inf = 1.f; }
    scaled_grad[i] = inv_scale_value == 1.f ? scaled_grad[i] : scaled_grad[i] * inv_scale_value;
  }
}

};  // namespace

template<typename T>
class AMPUpdateScaleGpuKernel final : public user_op::OpKernel {
 public:
  AMPUpdateScaleGpuKernel() = default;
  ~AMPUpdateScaleGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* current_scale = ctx->Tensor4ArgNameAndIndex("current_scale", 0);
    user_op::Tensor* growth_tracker = ctx->Tensor4ArgNameAndIndex("growth_tracker", 0);
    const user_op::Tensor* found_inf = ctx->Tensor4ArgNameAndIndex("found_inf", 0);

    const double growth_factor = ctx->Attr<double>("growth_factor");
    const double backoff_factor = ctx->Attr<double>("backoff_factor");
    const int64_t growth_interval = ctx->Attr<int64_t>("growth_interval");

    AmpUpdateScaleImpl<T><<<1, 1, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        current_scale->mut_dptr<T>(), growth_tracker->mut_dptr<int>(), found_inf->dptr<T>(),
        growth_factor, backoff_factor, growth_interval);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class AMPForEachNonFiniteCheckAndUnscaleGpuKernel final : public user_op::OpKernel {
 public:
  AMPForEachNonFiniteCheckAndUnscaleGpuKernel() = default;
  ~AMPForEachNonFiniteCheckAndUnscaleGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    size_t in_num = ctx->input_size("scaled_grads_found_inf_inv_scale");
    user_op::Tensor* found_inf =
        ctx->Tensor4ArgNameAndIndex("scaled_grads_found_inf_inv_scale", in_num - 2);
    const user_op::Tensor* inv_scale =
        ctx->Tensor4ArgNameAndIndex("scaled_grads_found_inf_inv_scale", in_num - 1);
    for (size_t i = 0; i < in_num - 2; ++i) {
      user_op::Tensor* scaled_grad =
          ctx->Tensor4ArgNameAndIndex("scaled_grads_found_inf_inv_scale", i);
      const int32_t elem_cnt = scaled_grad->shape_view().elem_cnt();
      RUN_CUDA_KERNEL((AmpForEachNonFiniteCheckAndUnscaleImpl<T>), ctx->stream(), elem_cnt,
                      elem_cnt, scaled_grad->mut_dptr<T>(), found_inf->mut_dptr<float>(),
                      inv_scale->dptr<float>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_AMP_UPDATE_SCALE_CUDA_KERNEL(dtype)      \
  REGISTER_USER_KERNEL("amp_update_scale")                \
      .SetCreateFn<AMPUpdateScaleGpuKernel<dtype>>()      \
      .SetIsMatchedHob(                                   \
          (user_op::HobDeviceType() == DeviceType::kCUDA) \
          && (user_op::HobDataType("current_scale", 0) == GetDataType<dtype>::value));

#define REGISTER_AMP_FOR_EACH_NONFINITE_CHECK_AND_UNSCALE_CUDA_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("amp_non_finite_check_and_unscale")                             \
      .SetCreateFn<AMPForEachNonFiniteCheckAndUnscaleGpuKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("scaled_grads_found_inf_inv_scale", 0) \
                           == GetDataType<dtype>::value));

REGISTER_AMP_UPDATE_SCALE_CUDA_KERNEL(float)
REGISTER_AMP_UPDATE_SCALE_CUDA_KERNEL(double)
REGISTER_AMP_FOR_EACH_NONFINITE_CHECK_AND_UNSCALE_CUDA_KERNEL(float)
REGISTER_AMP_FOR_EACH_NONFINITE_CHECK_AND_UNSCALE_CUDA_KERNEL(double)

namespace {

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

}  // namespace

template<typename T>
__global__ void MultiTensorAMPForEachNonFiniteCheckAndUnscaleGpu(
    int32_t num_tensor, float* found_inf, const float* inv_scale,
    TensorTupleParams tensor_tuple_params) {
  int64_t v_block_id = blockIdx.x;
  const auto inv_scale_value = *inv_scale;
  for (int64_t tensor_idx = 0; tensor_idx < num_tensor; tensor_idx++) {
    const int64_t tensor_elem_cnt = tensor_tuple_params.sizes[tensor_idx];
    T* scaled_grad = (T*)tensor_tuple_params.ptr[tensor_idx];

    for (int64_t i = v_block_id * blockDim.x * kUnrollSize + threadIdx.x; i < tensor_elem_cnt;
         i += blockDim.x * gridDim.x * kUnrollSize) {
      if (!IsFinite(scaled_grad[i])) { *found_inf = 1.f; }

#pragma unroll
      for (int32_t ilp = 0; ilp < kUnrollSize; ilp++) {
        int64_t actual_idx = i + ilp * blockDim.x;
        if (actual_idx < tensor_elem_cnt) {
          scaled_grad[i] =
              inv_scale_value == 1.f ? scaled_grad[i] : scaled_grad[i] * inv_scale_value;
        }
      }
    }
    v_block_id -= tensor_tuple_params.block_offset[tensor_idx];
    if (v_block_id < 0) { v_block_id += gridDim.x; }
  }
}

template<typename T>
class MultiTensorAMPForEachNonFiniteCheckAndUnscaleGpuKernel final : public user_op::OpKernel {
 public:
  MultiTensorAMPForEachNonFiniteCheckAndUnscaleGpuKernel() = default;
  ~MultiTensorAMPForEachNonFiniteCheckAndUnscaleGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    size_t in_num = ctx->input_size("scaled_grads_found_inf_inv_scale");
    user_op::Tensor* found_inf =
        ctx->Tensor4ArgNameAndIndex("scaled_grads_found_inf_inv_scale", in_num - 2);
    const user_op::Tensor* inv_scale =
        ctx->Tensor4ArgNameAndIndex("scaled_grads_found_inf_inv_scale", in_num - 1);

    TensorTupleParams tensor_tuple_params{};
    int32_t count = 0;
    int32_t total_elem_cnt = 0;
    for (int tensor_idx = 0; tensor_idx < in_num - 2; tensor_idx++) {
      user_op::Tensor* tensor =
          ctx->Tensor4ArgNameAndIndex("scaled_grads_found_inf_inv_scale", tensor_idx);
      tensor_tuple_params.ptr[count] = tensor->mut_dptr();

      const int64_t tensor_elem_cnt = tensor->shape_view().elem_cnt();
      tensor_tuple_params.sizes[count] = tensor_elem_cnt;

      count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (count == kMaxTuples || tensor_idx == in_num - 3) {
        const unsigned int grid_size =
            ComputeGridSize(ctx->stream()->As<ep::CudaStream>(), kBlockSize, tensor_elem_cnt);
        for (int i = 0; i < count; i++) {
          tensor_tuple_params.block_offset[i] =
              ((tensor_tuple_params.sizes[i] + kBlockSize * kUnrollSize - 1)
               / (kBlockSize * kUnrollSize))
              % grid_size;
        }
        MultiTensorAMPForEachNonFiniteCheckAndUnscaleGpu<T>
            <<<grid_size, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
                count, found_inf->mut_dptr<float>(), inv_scale->dptr<float>(), tensor_tuple_params);

        count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MULTI_TENSOR_AMP_FOR_EACH_NONFINITE_CHECK_AND_UNSCALE_CUDA_KERNEL(dtype) \
  REGISTER_USER_KERNEL("multi_tensor_amp_non_finite_check_and_unscale")                   \
      .SetCreateFn<MultiTensorAMPForEachNonFiniteCheckAndUnscaleGpuKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                    \
                       && (user_op::HobDataType("scaled_grads_found_inf_inv_scale", 0)    \
                           == GetDataType<dtype>::value));

REGISTER_MULTI_TENSOR_AMP_FOR_EACH_NONFINITE_CHECK_AND_UNSCALE_CUDA_KERNEL(float)
REGISTER_MULTI_TENSOR_AMP_FOR_EACH_NONFINITE_CHECK_AND_UNSCALE_CUDA_KERNEL(double)

}  // namespace oneflow
