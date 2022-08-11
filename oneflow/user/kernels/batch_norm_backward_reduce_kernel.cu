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
#include <limits>
#include <algorithm>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/cuda/cuda_device.h"
#include "oneflow/user/kernels/batch_norm_kernel_utils.h"

// NOTE(Liang Depeng):
// The implementation of batch_norm_backward_reduce kernel is modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Normalization.cuh

namespace oneflow {

namespace {

template<typename T>
static size_t InferTmpSizeForChannelLastKernel(user_op::InferContext* ctx) {
  const int32_t axis = ctx->Attr<int32_t>("axis");
  const Shape& in_shape = ctx->InputTensorDesc("input", 0).shape();
  const int64_t stride = in_shape.At(axis);
  const int64_t reduction_size = in_shape.elem_cnt() / stride;
  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid, true);
  size_t tmp_size = 0;
  if (grid.y > 1) {
    tmp_size += 2 * stride * grid.y * sizeof(T);
    tmp_size += grid.x * sizeof(int32_t);
  }
  return tmp_size;
}

template<typename T, typename ACC_T>
struct Float2 {
  ACC_T v1, v2;
  __device__ Float2() {}
  __device__ Float2(T v1, T v2) : v1(static_cast<ACC_T>(v1)), v2(static_cast<ACC_T>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<ACC_T>(v)), v2(static_cast<ACC_T>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

// Sum across all threads within a warp
template<typename T>
static __device__ __forceinline__ T warpSum_(T val) {
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) { val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE); }
  return val;
}

template<typename RES_T>
static __device__ __forceinline__ RES_T warpSum(RES_T value) {
  value.v1 = warpSum_(value.v1);
  value.v2 = warpSum_(value.v2);
  return value;
}

template<typename RES_T, typename T, typename ACC_T, typename IDX_TYPE>
__device__ RES_T reduce(const T* input_ptr, const T* grad_out_ptr, ACC_T r_mean, IDX_TYPE channel,
                        IDX_TYPE batch_size, IDX_TYPE channel_size, IDX_TYPE spatial_size) {
  IDX_TYPE batch_offset = spatial_size * channel_size;
  IDX_TYPE channel_offset = channel * spatial_size;
  // first the reductions each thread does separately
  RES_T sum = static_cast<RES_T>(0);
  for (int batch = threadIdx.y; batch < batch_size; batch += blockDim.y) {
    IDX_TYPE offset = batch * batch_offset;
    for (int x = threadIdx.x; x < spatial_size; x += blockDim.x) {
      //   sum += op(batch, plane, x);
      ACC_T g = grad_out_ptr[offset + channel_offset + x];
      ACC_T c = static_cast<ACC_T>(input_ptr[offset + channel_offset + x]) - r_mean;
      sum.v1 += g;
      sum.v2 += g * c;
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __shared__ RES_T shared[WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % WARP_SIZE == 0) { shared[tid / WARP_SIZE] = sum; }
  if (tid >= blockDim.x * blockDim.y / WARP_SIZE && tid < WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (RES_T)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) { shared[0] = sum; }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template<typename T, typename ACC_T, typename IDX_TYPE>
__global__ void batch_norm_backward_reduce_kernel(
    const IDX_TYPE batch_size, const IDX_TYPE channel_size, const IDX_TYPE spatial_size,
    const T* grad_out_ptr, const T* input_ptr, const T* mean_ptr, const T* invstd_ptr,
    T* sum_dy_ptr, T* sum_dy_xmu_ptr, T* grad_weight_ptr, T* grad_bias_ptr) {
  IDX_TYPE channel = blockIdx.x;
  ACC_T r_mean = mean_ptr[channel];
  ACC_T factor = invstd_ptr[channel];

  auto res = reduce<Float2<T, ACC_T>, T, ACC_T, IDX_TYPE>(input_ptr, grad_out_ptr, r_mean, channel,
                                                          batch_size, channel_size, spatial_size);

  if (threadIdx.x == 0) {
    if (grad_weight_ptr != nullptr) { grad_weight_ptr[channel] = static_cast<T>(res.v2 * factor); }
    if (grad_bias_ptr != nullptr) { grad_bias_ptr[channel] = static_cast<T>(res.v1); }
    if (sum_dy_ptr != nullptr) { sum_dy_ptr[channel] = static_cast<ACC_T>(res.v1); }
    if (sum_dy_xmu_ptr != nullptr) { sum_dy_xmu_ptr[channel] = static_cast<ACC_T>(res.v2); }
  }
}

template<typename T>
__device__ __forceinline__ void merge_block_vertical_backward(T& sum_dy, T& sum_dy_xmu,
                                                              T* shmem_sum_dy,
                                                              T* shmem_sum_dy_xmu) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset * 2) {
      shmem_sum_dy[address_base] = sum_dy;
      shmem_sum_dy_xmu[address_base] = sum_dy_xmu;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;

      sum_dy += shmem_sum_dy[address];
      sum_dy_xmu += shmem_sum_dy_xmu[address];
    }
  }
}

template<typename T, typename ACC_T, typename IDX_TYPE, int PARALLEL_LOADS>
__global__ void batch_norm_backward_reduce_channels_last_kernel(
    const T* __restrict__ grad_output_ptr, const T* __restrict__ input_ptr,
    const ACC_T* __restrict__ mean_ptr, const ACC_T* __restrict__ inv_std_ptr,
    ACC_T* __restrict__ sum_dy_o_ptr, ACC_T* __restrict__ sum_dy_xmu_o_ptr,
    T* __restrict__ grad_weight_ptr, T* __restrict__ grad_bias_ptr,
    volatile ACC_T* staging_data_ptr, int32_t* semaphores_ptr, const IDX_TYPE reduction_size,
    const IDX_TYPE stride) {
  // hide latency with concurrency
  ACC_T sum_dy[PARALLEL_LOADS];
  ACC_T sum_dy_xmu[PARALLEL_LOADS];

#pragma unroll
  for (int i = 0; i < PARALLEL_LOADS; i++) {
    sum_dy[i] = ACC_T(0);
    sum_dy_xmu[i] = ACC_T(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (c_offset >= stride || m_offset >= reduction_size) { return; }

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  auto r_mean = mean_ptr[c_offset];
  auto factor = inv_std_ptr[c_offset];

  for (int i = 0; i < loop_count; i++) {
    ACC_T x_input[PARALLEL_LOADS];
    ACC_T x_grad_output[PARALLEL_LOADS];

    // load multiple data in
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_input[j] = input_ptr[address_base];
        x_grad_output[j] = grad_output_ptr[address_base];
      } else {
        x_input[j] = ACC_T(0);
        x_grad_output[j] = ACC_T(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

    // calculate sum_dy / sum_dy_xmu
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      sum_dy[j] += x_grad_output[j];
      sum_dy_xmu[j] += x_grad_output[j] * (x_input[j] - r_mean);
    }
  }

  // thread reduction to accumulate sum_dy / sum_dy_xmu between PARALLEL_LOADS
#pragma unroll
  for (int j = 1; j < PARALLEL_LOADS; j++) {
    sum_dy[0] += sum_dy[j];
    sum_dy_xmu[0] += sum_dy_xmu[j];
  }

  // release array of registers
  auto sum_dy_th = sum_dy[0];
  auto sum_dy_xmu_th = sum_dy_xmu[0];

  // block-wise reduction with shared memory (since reduction cannot be done within a warp)
  static __shared__ ACC_T shmem_sum_dy[MAX_BLOCK_SIZE];
  static __shared__ ACC_T shmem_sum_dy_xmu[MAX_BLOCK_SIZE];

  merge_block_vertical_backward(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);

  if (gridDim.y > 1) {
    volatile ACC_T* staging_sum_dy = staging_data_ptr;
    volatile ACC_T* staging_sum_dy_xmu = &staging_data_ptr[stride * gridDim.y];

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_sum_dy[address_base] = sum_dy_th;
      staging_sum_dy_xmu[address_base] = sum_dy_xmu_th;
    }

    __threadfence();
    __syncthreads();  // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores_ptr[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y - 1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      sum_dy_th = ACC_T(0.0);
      sum_dy_xmu_th = ACC_T(0.0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        sum_dy_th += (c_offset < stride ? staging_sum_dy[address_base] : ACC_T(0.0));
        sum_dy_xmu_th += (c_offset < stride ? staging_sum_dy_xmu[address_base] : ACC_T(0.0));
      }

      merge_block_vertical_backward(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);
      if (threadIdx.y == 0 && c_offset < stride) {
        if (grad_bias_ptr != nullptr) { grad_bias_ptr[c_offset] = static_cast<T>(sum_dy_th); }
        if (grad_weight_ptr != nullptr) {
          grad_weight_ptr[c_offset] = static_cast<T>(sum_dy_xmu_th * factor);
        }
        sum_dy_o_ptr[c_offset] = sum_dy_th;
        sum_dy_xmu_o_ptr[c_offset] = sum_dy_xmu_th;
      }
    }
  } else {
    if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
      if (grad_bias_ptr != nullptr) { grad_bias_ptr[c_offset] = static_cast<T>(sum_dy_th); }
      if (grad_weight_ptr != nullptr) {
        grad_weight_ptr[c_offset] = static_cast<T>(sum_dy_xmu_th * factor);
      }
      sum_dy_o_ptr[c_offset] = sum_dy_th;
      sum_dy_xmu_o_ptr[c_offset] = sum_dy_xmu_th;
    }
  }
}

template<typename T>
struct BatchNormBackwardReduceFunctor final {
  void operator()(ep::Stream* stream, const int64_t batch_size, const int64_t channel_size,
                  const int64_t spatial_size, const T* grad_out_ptr, const T* input_ptr,
                  const T* mean_ptr, const T* invstd_ptr, T* sum_dy_ptr, T* sum_dy_xmu_ptr,
                  T* grad_weight_ptr, T* grad_bias_ptr) {
    using ACC_T = acc_type<T>;
    int block_y = std::min<int>(lastPow2(batch_size), MAX_BLOCK_SIZE / WARP_SIZE);
    // We want block_x to be at least a warp width
    int block_x = std::min<int>(std::max<int>(getNumThreads(spatial_size), WARP_SIZE),
                                MAX_BLOCK_SIZE / block_y);
    const dim3 block(block_x, block_y);
    const dim3 grid(channel_size);

    if (batch_size * channel_size * spatial_size < std::numeric_limits<int32_t>::max()) {
      batch_norm_backward_reduce_kernel<T, ACC_T, int32_t>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              static_cast<int32_t>(batch_size), static_cast<int32_t>(channel_size),
              static_cast<int32_t>(spatial_size), grad_out_ptr, input_ptr, mean_ptr, invstd_ptr,
              sum_dy_ptr, sum_dy_xmu_ptr, grad_weight_ptr, grad_bias_ptr);
    } else {
      batch_norm_backward_reduce_kernel<T, ACC_T, int64_t>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              batch_size, channel_size, spatial_size, grad_out_ptr, input_ptr, mean_ptr, invstd_ptr,
              sum_dy_ptr, sum_dy_xmu_ptr, grad_weight_ptr, grad_bias_ptr);
    }
  }
};

template<typename T>
struct BatchNormBackwardReduceChannelLastFunctor final {
  void operator()(ep::Stream* stream, const int64_t stride, const int64_t reduction_size,
                  const T* grad_out_ptr, const T* input_ptr, const T* mean_ptr, const T* invstd_ptr,
                  T* sum_dy_ptr, T* sum_dy_xmu_ptr, T* grad_weight_ptr, T* grad_bias_ptr,
                  user_op::Tensor* tmp_buffer) {
    using ACC_T = acc_type<T>;

    dim3 block;
    dim3 grid;
    flexible_launch_configs(reduction_size, stride, block, grid, true);

    T* staging_data_ptr = nullptr;
    int32_t* semaphores_ptr = nullptr;
    if (grid.y > 1) {
      staging_data_ptr = tmp_buffer->mut_dptr<T>();
      semaphores_ptr = reinterpret_cast<int32_t*>(tmp_buffer->mut_dptr<char>()
                                                  + 2 * stride * grid.y * sizeof(T));
    }

    if (stride * reduction_size < std::numeric_limits<int32_t>::max()) {
      batch_norm_backward_reduce_channels_last_kernel<T, ACC_T, int32_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              grad_out_ptr, input_ptr, mean_ptr, invstd_ptr, sum_dy_ptr, sum_dy_xmu_ptr,
              grad_weight_ptr, grad_bias_ptr, staging_data_ptr, semaphores_ptr,
              static_cast<int32_t>(reduction_size), static_cast<int32_t>(stride));
    } else {
      batch_norm_backward_reduce_channels_last_kernel<T, ACC_T, int64_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              grad_out_ptr, input_ptr, mean_ptr, invstd_ptr, sum_dy_ptr, sum_dy_xmu_ptr,
              grad_weight_ptr, grad_bias_ptr, staging_data_ptr, semaphores_ptr, reduction_size,
              stride);
    }
  }
};

}  // namespace

template<typename T>
class GpuBatchNormBackwardReduceKernel final : public user_op::OpKernel {
 public:
  GpuBatchNormBackwardReduceKernel() = default;
  ~GpuBatchNormBackwardReduceKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* grad_out = ctx->Tensor4ArgNameAndIndex("grad_out", 0);
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* invstd = ctx->Tensor4ArgNameAndIndex("invstd", 0);

    user_op::Tensor* sum_dy = ctx->Tensor4ArgNameAndIndex("sum_dy", 0);
    user_op::Tensor* sum_dy_xmu = ctx->Tensor4ArgNameAndIndex("sum_dy_xmu", 0);
    user_op::Tensor* grad_weight = ctx->Tensor4ArgNameAndIndex("grad_weight", 0);
    user_op::Tensor* grad_bias = ctx->Tensor4ArgNameAndIndex("grad_bias", 0);

    const T* grad_out_ptr = grad_out->dptr<T>();
    const T* input_ptr = input->dptr<T>();
    const T* mean_ptr = mean->dptr<T>();
    const T* invstd_ptr = invstd->dptr<T>();

    T* sum_dy_ptr = sum_dy->mut_dptr<T>();
    T* sum_dy_xmu_ptr = sum_dy_xmu->mut_dptr<T>();
    T* grad_weight_ptr = grad_weight->mut_dptr<T>();
    T* grad_bias_ptr = grad_bias->mut_dptr<T>();

    const int32_t axis = ctx->Attr<int32_t>("axis");

    bool use_channels_last_kernel = axis == 1 ? false : true;
    if (use_channels_last_kernel) {  // NHWC format
      const int64_t stride = input->shape_view().At(axis);
      const int64_t reduction_size = input->shape_view().elem_cnt() / stride;
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      BatchNormBackwardReduceChannelLastFunctor<T>()(
          ctx->stream(), stride, reduction_size, grad_out_ptr, input_ptr, mean_ptr, invstd_ptr,
          sum_dy_ptr, sum_dy_xmu_ptr, grad_weight_ptr, grad_bias_ptr, tmp_buffer);
    } else {  // NCHW format
      const int64_t batch_size = input->shape_view().At(0);
      const int64_t channel_size = input->shape_view().At(1);
      const int64_t spatial_size = input->shape_view().Count(2);

      BatchNormBackwardReduceFunctor<T>()(ctx->stream(), batch_size, channel_size, spatial_size,
                                          grad_out_ptr, input_ptr, mean_ptr, invstd_ptr, sum_dy_ptr,
                                          sum_dy_xmu_ptr, grad_weight_ptr, grad_bias_ptr);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BATCH_NORM_BACKWARD_REDUCE_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("batch_norm_backward_reduce")                                         \
      .SetCreateFn<GpuBatchNormBackwardReduceKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                       \
                       && (user_op::HobDataType("grad_out", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)    \
                       && (user_op::HobDataType("mean", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("invstd", 0) == GetDataType<dtype>::value))  \
      .SetInferTmpSizeFn(InferTmpSizeForChannelLastKernel<dtype>)

REGISTER_BATCH_NORM_BACKWARD_REDUCE_KERNEL(float);
REGISTER_BATCH_NORM_BACKWARD_REDUCE_KERNEL(double);

}  // namespace oneflow
