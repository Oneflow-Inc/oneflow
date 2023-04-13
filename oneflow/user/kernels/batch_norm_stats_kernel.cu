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
// The implementation of batch_norm_stats kernel is modified from
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
    tmp_size += 4 * stride * grid.y * sizeof(T);
    tmp_size += grid.x * sizeof(int32_t);
  }
  return tmp_size;
}

template<typename T, typename C>
__device__ __forceinline__ void welford_merge_element(C& count, T& mean, T& m2n, const C& count_new,
                                                      const T& mean_new, const T& m2n_new) {
  T factor = T(1.0) / ::max(C(1), (count + count_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * count_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
  count += count_new;
}

// merge mean/m2n among threadIdx.y within block
template<typename T, typename C>
__device__ __forceinline__ void welford_merge_block_vertical(C& count, T& mean, T& m2n,
                                                             C* shmem_count, T* shmem_mean,
                                                             T* shmem_m2n) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset * 2) {
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;
      // read shared memory back to register for reduction
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];

      welford_merge_element(count, mean, m2n, count_new, mean_new, m2n_new);
    }
  }
}

template<typename T, typename ACC_T, typename IDX_TYPE, int PARALLEL_LOADS>
__global__ void batch_norm_collect_statistics_channels_last_kernel(
    const T* __restrict__ input_ptr, ACC_T* __restrict__ out_mean_ptr,
    ACC_T* __restrict__ out_invstd_ptr, volatile ACC_T* staging_data_ptr, int32_t* semaphores_ptr,
    const IDX_TYPE reduction_size, const IDX_TYPE stride, ACC_T epsilon) {
  // hide latency with concurrency
  ACC_T x_mean[PARALLEL_LOADS];
  ACC_T m_2_n[PARALLEL_LOADS];
  IDX_TYPE count[PARALLEL_LOADS];

#pragma unroll
  for (IDX_TYPE i = 0; i < PARALLEL_LOADS; i++) {
    x_mean[i] = ACC_T(0);
    m_2_n[i] = ACC_T(0);
    count[i] = ACC_T(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  IDX_TYPE inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  IDX_TYPE m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  IDX_TYPE c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  IDX_TYPE loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  IDX_TYPE address_base = m_offset * stride + c_offset;
  IDX_TYPE address_increment = inner_loop_stride * stride;

  for (IDX_TYPE i = 0; i < loop_count; i++) {
    ACC_T x_math[PARALLEL_LOADS];
    ACC_T x_count_inv[PARALLEL_LOADS];
    ACC_T is_valid[PARALLEL_LOADS];

    // load multiple data in
#pragma unroll
    for (IDX_TYPE j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_math[j] = input_ptr[address_base];
        count[j]++;
        x_count_inv[j] = ACC_T(1) / count[j];
        is_valid[j] = ACC_T(1);
      } else {
        x_math[j] = ACC_T(0);
        x_count_inv[j] = ACC_T(0);
        is_valid[j] = ACC_T(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

    // calculate mean/m2n with welford
#pragma unroll
    for (IDX_TYPE j = 0; j < PARALLEL_LOADS; j++) {
      ACC_T delta0 = x_math[j] - x_mean[j];
      x_mean[j] += delta0 * x_count_inv[j];
      ACC_T delta1 = x_math[j] - x_mean[j];
      m_2_n[j] += delta0 * delta1 * is_valid[j];
    }
  }

  // thread reduction to accumulate mean/m_2_n/count between PARALLEL_LOADS
#pragma unroll
  for (IDX_TYPE j = 1; j < PARALLEL_LOADS; j++) {
    welford_merge_element(count[0], x_mean[0], m_2_n[0], count[j], x_mean[j], m_2_n[j]);
  }

  // release x_mean / m_2_n
  auto mean_th = x_mean[0];
  auto m2_th = m_2_n[0];
  auto count_th = count[0];

  // block-wise reduction with shared memory (since reduction cannot be done within a warp)
  static __shared__ ACC_T shmem_mean[MAX_BLOCK_SIZE];
  static __shared__ ACC_T shmem_m2n[MAX_BLOCK_SIZE];
  static __shared__ IDX_TYPE shmem_count[MAX_BLOCK_SIZE];

  welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count, shmem_mean, shmem_m2n);

  if (gridDim.y > 1) {
    volatile ACC_T* staging_mean = staging_data_ptr;
    volatile ACC_T* staging_m2n = &staging_data_ptr[stride * gridDim.y];
    volatile IDX_TYPE* staging_count =
        reinterpret_cast<volatile IDX_TYPE*>(&staging_m2n[stride * gridDim.y]);

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data_ptr;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_mean[address_base] = mean_th;
      staging_m2n[address_base] = m2_th;
      staging_count[address_base] = count_th;
    }

    __threadfence();
    __syncthreads();  // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      IDX_TYPE old = atomicAdd(&semaphores_ptr[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y - 1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      count_th = 0;
      mean_th = ACC_T(0.0);
      m2_th = ACC_T(0.0);

      for (IDX_TYPE y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        IDX_TYPE count_new = c_offset < stride ? staging_count[address_base] : 0;
        ACC_T mean_new = c_offset < stride ? staging_mean[address_base] : ACC_T(0.0);
        ACC_T m2n_new = c_offset < stride ? staging_m2n[address_base] : ACC_T(0.0);

        welford_merge_element(count_th, mean_th, m2_th, count_new, mean_new, m2n_new);
      }

      welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count, shmem_mean, shmem_m2n);
      if (threadIdx.y == 0 && c_offset < stride) {
        out_mean_ptr[c_offset] = static_cast<ACC_T>(mean_th);
        out_invstd_ptr[c_offset] = inv_std(m2_th / count_th, epsilon);
      }
    }
  } else {
    if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
      out_mean_ptr[c_offset] = static_cast<ACC_T>(mean_th);
      out_invstd_ptr[c_offset] = inv_std(m2_th / count_th, epsilon);
    }
  }
}

template<typename T, typename ACC_T, typename IDX_TYPE>
__global__ void batch_norm_collect_statistics_kernel(const T* input_ptr, const IDX_TYPE batch_size,
                                                     const IDX_TYPE channel_size,
                                                     const IDX_TYPE spatial_size, const ACC_T eps,
                                                     T* mean_ptr, T* invstd_ptr) {
  __shared__ IDX_TYPE shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];

  IDX_TYPE channel_idx = blockIdx.x;
  IDX_TYPE N = batch_size * spatial_size;
  IDX_TYPE tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  ACC_T* shared_avg_var = (ACC_T*)&shared_n[WARP_SIZE];

  // first the reductions each thread does separately
  ACC_T avg = 0;
  ACC_T var_n = 0;
  IDX_TYPE n = 0;
  const IDX_TYPE channel_offset = channel_idx * spatial_size;
  const IDX_TYPE batch_offset = channel_size * spatial_size;
  for (IDX_TYPE batch = threadIdx.y; batch < batch_size; batch += blockDim.y) {
    IDX_TYPE offset = batch * batch_offset + channel_offset;
    for (IDX_TYPE x = threadIdx.x; x < spatial_size; x += blockDim.x) {
      ACC_T v = input_ptr[offset + x];
      ACC_T d1 = v - avg;
      n++;
      avg += d1 / n;
      var_n += d1 * (v - avg);
    }
  }

  // summing the result of all the threads within a warp
  // refer to: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  // first warpSum to get one value per thread to one value per warp
  for (IDX_TYPE i = 0; i < getMSB(WARP_SIZE); ++i) {
    ACC_T o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    IDX_TYPE o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    ACC_T factor = 1.0 / fmaxf(1.0, n + o_n);
    var_n +=
        WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warp's final sum result into shared memory
  // there are at most (thread_number_of_a_block / WARP_SIZE) results
  __syncthreads();
  if (tid % WARP_SIZE == 0) {
    shared_n[tid / WARP_SIZE] = n;
    shared_avg_var[tid / WARP_SIZE * 2] = avg;
    shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();

  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.
  if (tid < WARP_SIZE) {
    // initialize n, avg and var_n of each thread within the first warp
    n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_avg_var[2 * tid] : ACC_T(0));
    var_n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_avg_var[2 * tid + 1] : ACC_T(0));

    for (IDX_TYPE i = 0; i < getMSB(WARP_SIZE); ++i) {
      ACC_T o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
      IDX_TYPE o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
      ACC_T factor = 1.0 / fmaxf(1.0, n + o_n);
      var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE)
               + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
      avg = (n * avg + o_n * o_avg) * factor;
      n += o_n;
    }
  }

  // save the mean and inverse standard deviation
  if (tid == 0) {
    mean_ptr[channel_idx] = avg;
    invstd_ptr[channel_idx] = inv_std(var_n / N, eps);
  }
}

template<typename T>
struct BatchNormStatsFunctor final {
  void operator()(ep::Stream* stream, const user_op::Tensor* input, user_op::Tensor* mean,
                  user_op::Tensor* invstd, const float eps) {
    using ACC_T = acc_type<T>;
    const ShapeView& input_shape = input->shape_view();
    const int64_t input_numel = input_shape.elem_cnt();
    const int64_t spatial_size = input_shape.Count(2);

    dim3 blocks(input_shape.At(1));
    int32_t tf = getNumThreads(spatial_size);
    dim3 threads(tf, std::max<int32_t>(1, MAX_BLOCK_SIZE / tf));

    const T* input_ptr = input->dptr<T>();
    T* mean_ptr = mean->mut_dptr<T>();
    T* invstd_ptr = invstd->mut_dptr<T>();

    if (input_numel < std::numeric_limits<int32_t>::max()) {
      batch_norm_collect_statistics_kernel<T, ACC_T, int32_t>
          <<<blocks, threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              input_ptr, static_cast<int32_t>(input_shape.At(0)),
              static_cast<int32_t>(input_shape.At(1)), static_cast<int32_t>(spatial_size), eps,
              mean_ptr, invstd_ptr);
    } else {
      batch_norm_collect_statistics_kernel<T, ACC_T, int64_t>
          <<<blocks, threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              input_ptr, input_shape.At(0), input_shape.At(1), spatial_size, eps, mean_ptr,
              invstd_ptr);
    }
  }
};

template<typename T>
struct BatchNormStatsChannelLastFunctor final {
  void operator()(ep::Stream* stream, const user_op::Tensor* input, user_op::Tensor* mean,
                  user_op::Tensor* invstd, user_op::Tensor* tmp_buffer, const float eps,
                  const int32_t axis) {
    using ACC_T = acc_type<T>;
    const ShapeView& input_shape = input->shape_view();
    const int64_t stride = input_shape.At(axis);
    const int64_t reduction_size = input_shape.elem_cnt() / stride;

    dim3 block;
    dim3 grid;
    flexible_launch_configs(reduction_size, stride, block, grid, true);

    T* staging_data_ptr = nullptr;
    int32_t* semaphores_ptr = nullptr;
    if (grid.y > 1) {
      staging_data_ptr = tmp_buffer->mut_dptr<T>();
      semaphores_ptr = reinterpret_cast<int32_t*>(tmp_buffer->mut_dptr<char>()
                                                  + 4 * stride * grid.y * sizeof(T));
    }

    const T* input_ptr = input->dptr<T>();
    T* mean_ptr = mean->mut_dptr<T>();
    T* invstd_ptr = invstd->mut_dptr<T>();

    if (input_shape.elem_cnt() < std::numeric_limits<int32_t>::max()) {
      batch_norm_collect_statistics_channels_last_kernel<T, ACC_T, int32_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              input_ptr, mean_ptr, invstd_ptr, staging_data_ptr, semaphores_ptr,
              static_cast<int32_t>(reduction_size), static_cast<int32_t>(stride), eps);
    } else {
      batch_norm_collect_statistics_channels_last_kernel<T, ACC_T, int64_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              input_ptr, mean_ptr, invstd_ptr, staging_data_ptr, semaphores_ptr, reduction_size,
              stride, eps);
    }
  }
};

}  // namespace

template<typename T>
class GpuBatchNormStatsKernel final : public user_op::OpKernel {
 public:
  GpuBatchNormStatsKernel() = default;
  ~GpuBatchNormStatsKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* invstd = ctx->Tensor4ArgNameAndIndex("invstd", 0);

    const int32_t axis = ctx->Attr<int32_t>("axis");
    const float eps = ctx->Attr<float>("eps");

    bool use_channels_last_kernel = axis == 1 ? false : true;
    if (use_channels_last_kernel) {  // NHWC format
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      BatchNormStatsChannelLastFunctor<T>()(ctx->stream(), input, mean, invstd, tmp_buffer, eps,
                                            axis);
    } else {  // NCHW format
      BatchNormStatsFunctor<T>()(ctx->stream(), input, mean, invstd, eps);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BATCH_NORM_STATS_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("batch_norm_stats")                                                 \
      .SetCreateFn<GpuBatchNormStatsKernel<dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTmpSizeForChannelLastKernel<dtype>)

REGISTER_BATCH_NORM_STATS_KERNEL(float);
REGISTER_BATCH_NORM_STATS_KERNEL(double);

}  // namespace oneflow
