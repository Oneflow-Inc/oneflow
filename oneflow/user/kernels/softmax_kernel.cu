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
#include "oneflow/user/kernels/softmax_kernel_util.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

constexpr int64_t kSoftmaxGpuBlockSize = 256;

template<typename T>
struct SoftmaxUtil {
  using ComputeType = T;
  __device__ static ComputeType ToComputeType(T v) { return v; }
  __device__ static T FromComputeType(ComputeType v) { return v; }
};

template<>
struct SoftmaxUtil<half> {
  using ComputeType = float;
  __device__ static ComputeType ToComputeType(half v) { return __half2float(v); }
  __device__ static half FromComputeType(ComputeType v) { return __float2half(v); }
};

__device__ double Exp(double x) { return exp(x); }

__device__ float Exp(float x) { return expf(x); }

template<typename T>
int GetForwardDynamicSharedMemorySize(const int num_classes) {
  return num_classes * sizeof(typename SoftmaxUtil<T>::ComputeType);
}

template<typename T>
int GetBackwardDynamicSharedMemorySize(const int num_classes) {
  return 2 * num_classes * sizeof(typename SoftmaxUtil<T>::ComputeType);
}

int GetSoftmaxBlockSize() { return kSoftmaxGpuBlockSize; }

int GetSoftmaxNumBlocks(const int num_instances) {
  return std::min(static_cast<int>(num_instances), kCudaMaxBlocksNum);
}

template<typename T>
int GetMinNumClasses() {
  return 32;
}

template<typename T>
__global__ void SoftmaxGpuForwardImpl(const int num_instances, const int num_classes, const T* in,
                                      T* prob) {
  using SU = SoftmaxUtil<T>;
  using ComputeType = typename SU::ComputeType;
  extern __shared__ __align__(sizeof(ComputeType)) unsigned char fw_shared_buf[];
  auto* compute_buf = reinterpret_cast<ComputeType*>(fw_shared_buf);
  __shared__ ComputeType row_reduce_result;
  typedef cub::BlockReduce<ComputeType, kSoftmaxGpuBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < num_instances; row += gridDim.x) {
    const int row_offset = row * num_classes;
    const T* in_row = in + row_offset;
    T* prob_row = prob + row_offset;
    ComputeType thread_max = GetMinVal<ComputeType>();
    for (int col = tid; col < num_classes; col += kSoftmaxGpuBlockSize) {
      const ComputeType x = SU::ToComputeType(in_row[col]);
      compute_buf[col] = x;
      thread_max = max(thread_max, x);
    }
    __syncthreads();
    ComputeType block_max = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_max, cub::Max());
    if (tid == 0) { row_reduce_result = block_max; }
    __syncthreads();
    const ComputeType row_max_t = row_reduce_result;
    ComputeType thread_sum = 0;
    for (int col = tid; col < num_classes; col += kSoftmaxGpuBlockSize) {
      const ComputeType exp_x = Exp(compute_buf[col] - row_max_t);
      compute_buf[col] = exp_x;
      thread_sum += exp_x;
    }
    __syncthreads();
    ComputeType block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_sum, cub::Sum());
    if (tid == 0) { row_reduce_result = block_sum; }
    __syncthreads();
    const ComputeType row_sum_t = row_reduce_result;
    for (int col = tid; col < num_classes; col += kSoftmaxGpuBlockSize) {
      prob_row[col] = SU::FromComputeType(compute_buf[col] / row_sum_t);
    }
  }
}

template<typename T>
void SoftmaxForwardGpu(DeviceCtx* ctx, const int num_instances, const int num_classes, const T* in,
                       T* prob) {
  SoftmaxGpuForwardImpl<<<GetSoftmaxNumBlocks(num_instances), GetSoftmaxBlockSize(),
                          GetForwardDynamicSharedMemorySize<T>(num_classes), ctx->cuda_stream()>>>(
      num_instances, num_classes, in, prob);
}

template<>
void SoftmaxForwardGpu<float16>(DeviceCtx* ctx, const int num_instances, const int num_classes,
                                const float16* in, float16* prob) {
  SoftmaxForwardGpu<half>(ctx, num_instances, num_classes, reinterpret_cast<const half*>(in),
                          reinterpret_cast<half*>(prob));
}

template<typename T>
int GetForwardFusedKernelMaxActiveBlocks(const int num_classes) {
  int max_active_blocks;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, SoftmaxGpuForwardImpl<T>, GetSoftmaxBlockSize(),
      GetForwardDynamicSharedMemorySize<T>(num_classes)));
  return max_active_blocks;
}

template<>
int GetForwardFusedKernelMaxActiveBlocks<float16>(const int num_classes) {
  int max_active_blocks;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, SoftmaxGpuForwardImpl<half>, GetSoftmaxBlockSize(),
      GetForwardDynamicSharedMemorySize<half>(num_classes)));
  return max_active_blocks;
}

template<typename T>
bool IsForwardFusedKernelSupported(const int num_classes) {
  if (num_classes >= GetMinNumClasses<T>()
      && GetForwardFusedKernelMaxActiveBlocks<T>(num_classes) > 0) {
    return true;
  } else {
    return false;
  }
}

template<typename T>
__global__ void SoftmaxGpuBackwardImpl(const int num_instances, const int num_classes, const T* dy,
                                       const T* prob, T* dx) {
  using SU = SoftmaxUtil<T>;
  using ComputeType = typename SU::ComputeType;
  extern __shared__ __align__(sizeof(ComputeType)) unsigned char bw_shared_buf[];
  auto* dy_buf = reinterpret_cast<ComputeType*>(bw_shared_buf);
  auto* prob_buf =
      reinterpret_cast<ComputeType*>(bw_shared_buf + num_classes * sizeof(ComputeType));
  __shared__ ComputeType row_reduce_result;
  typedef cub::BlockReduce<ComputeType, kSoftmaxGpuBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < num_instances; row += gridDim.x) {
    const int row_offset = row * num_classes;
    const T* dy_row = dy + row_offset;
    const T* prob_row = prob + row_offset;
    T* dx_row = dx + row_offset;
    ComputeType thread_sum = 0;
    for (int col = tid; col < num_classes; col += kSoftmaxGpuBlockSize) {
      const ComputeType dy_col = SU::ToComputeType(dy_row[col]);
      dy_buf[col] = dy_col;
      const ComputeType prob_col = SU::ToComputeType(prob_row[col]);
      prob_buf[col] = prob_col;
      thread_sum += (dy_col * prob_col);
    }
    __syncthreads();
    ComputeType block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_sum, cub::Sum());
    if (tid == 0) { row_reduce_result = block_sum; }
    __syncthreads();
    const ComputeType row_sum_t = row_reduce_result;
    for (int col = tid; col < num_classes; col += kSoftmaxGpuBlockSize) {
      dx_row[col] = SU::FromComputeType((dy_buf[col] - row_sum_t) * prob_buf[col]);
    }
  }
}

template<typename T>
void SoftmaxBackwardGpu(DeviceCtx* ctx, const int num_instances, const int num_classes, const T* in,
                        const T* prob, T* dx) {
  SoftmaxGpuBackwardImpl<<<GetSoftmaxNumBlocks(num_instances), GetSoftmaxBlockSize(),
                           GetBackwardDynamicSharedMemorySize<T>(num_classes),
                           ctx->cuda_stream()>>>(num_instances, num_classes, in, prob, dx);
}

template<>
void SoftmaxBackwardGpu<float16>(DeviceCtx* ctx, const int num_instances, const int num_classes,
                                 const float16* in, const float16* prob, float16* dx) {
  SoftmaxBackwardGpu<half>(ctx, num_instances, num_classes, reinterpret_cast<const half*>(in),
                           reinterpret_cast<const half*>(prob), reinterpret_cast<half*>(dx));
}

template<typename T>
int GetBackwardFusedKernelMaxActiveBlocks(const int num_classes) {
  int max_active_blocks;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, SoftmaxGpuBackwardImpl<T>, GetSoftmaxBlockSize(),
      GetBackwardDynamicSharedMemorySize<T>(num_classes)));
  return max_active_blocks;
}

template<>
int GetBackwardFusedKernelMaxActiveBlocks<float16>(const int num_classes) {
  int max_active_blocks;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, SoftmaxGpuBackwardImpl<half>, GetSoftmaxBlockSize(),
      GetBackwardDynamicSharedMemorySize<half>(num_classes)));
  return max_active_blocks;
}

template<typename T>
bool IsBackwardFusedKernelSupported(const int num_classes) {
  if (num_classes >= GetMinNumClasses<T>()
      && GetBackwardFusedKernelMaxActiveBlocks<T>(num_classes) > 0) {
    return true;
  } else {
    return false;
  }
}

template<typename T>
class SoftmaxKernel final : public user_op::OpKernel {
 public:
  SoftmaxKernel() = default;
  ~SoftmaxKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape();
    const int64_t num_classes = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t num_instances = in_shape.Count(0, in_shape.NumAxes() - 1);
    if (IsForwardFusedKernelSupported<T>(num_classes)) {
      SoftmaxForwardGpu<T>(ctx->device_ctx(), num_instances, num_classes, in->dptr<T>(),
                           out->mut_dptr<T>());
    } else {
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      SoftmaxKernelUtil<DeviceType::kGPU, T>::ComputeProb(
          ctx->device_ctx(), num_instances, num_classes, in->dptr<T>(), out->mut_dptr<T>(),
          tmp_buffer->mut_dptr(), tmp_buffer->shape().elem_cnt());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GPU_KERNEL(dtype)                                                       \
  REGISTER_USER_KERNEL("softmax")                                                                \
      .SetCreateFn<SoftmaxKernel<dtype>>()                                                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                             \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))          \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);                             \
        const int64_t num_classes = in_shape->At(in_shape->NumAxes() - 1);                       \
        const int64_t num_instances = in_shape->Count(0, in_shape->NumAxes() - 1);               \
        return SoftmaxKernelUtil<DeviceType::kGPU, dtype>::GetComputeProbTempStorageSizeInBytes( \
            num_instances, num_classes);                                                         \
      });

REGISTER_SOFTMAX_GPU_KERNEL(float16)
REGISTER_SOFTMAX_GPU_KERNEL(float)
REGISTER_SOFTMAX_GPU_KERNEL(double)
#undef REGISTER_SOFTMAX_GPU_KERNEL

template<typename T>
class SoftmaxGradKernel final : public user_op::OpKernel {
 public:
  SoftmaxGradKernel() = default;
  ~SoftmaxGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t num_classes = y->shape().At(y->shape().NumAxes() - 1);
    const int64_t num_instances = y->shape().elem_cnt() / num_classes;
    if (IsBackwardFusedKernelSupported<T>(num_classes)) {
      SoftmaxBackwardGpu<T>(ctx->device_ctx(), num_instances, num_classes, dy->dptr<T>(),
                            y->dptr<T>(), dx->mut_dptr<T>());
    } else {
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      SoftmaxKernelUtil<DeviceType::kGPU, T>::ComputeDiff(
          ctx->device_ctx(), num_instances, num_classes, dy->dptr<T>(), y->dptr<T>(),
          dx->mut_dptr<T>(), tmp_buffer->mut_dptr(), tmp_buffer->shape().elem_cnt());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GRAD_KERNEL(dtype)                                                      \
  REGISTER_USER_KERNEL("softmax_grad")                                                           \
      .SetCreateFn<SoftmaxGradKernel<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                             \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))           \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);                             \
        const int64_t num_classes = dy_shape->At(dy_shape->NumAxes() - 1);                       \
        const int64_t num_instances = dy_shape->Count(0, dy_shape->NumAxes() - 1);               \
        return SoftmaxKernelUtil<DeviceType::kGPU, dtype>::GetComputeProbTempStorageSizeInBytes( \
            num_instances, num_classes);                                                         \
      });

REGISTER_SOFTMAX_GRAD_KERNEL(float16)
REGISTER_SOFTMAX_GRAD_KERNEL(float)
REGISTER_SOFTMAX_GRAD_KERNEL(double)
#undef REGISTER_SOFTMAX_GRAD_KERNEL

}  // namespace

}  // namespace oneflow
