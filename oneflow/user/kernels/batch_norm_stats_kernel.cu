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

    int64_t spatial_size = 1;
    for (int64_t i = 2; i < input_shape.NumAxes(); ++i) { spatial_size *= input_shape.At(i); }

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

    } else {  // NCHW format
      BatchNormStatsFunctor<T>()(ctx->stream(), input, mean, invstd, eps);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BATCH_NORM_STATS_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("batch_norm_stats")                             \
      .SetCreateFn<GpuBatchNormStatsKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value))

REGISTER_BATCH_NORM_STATS_KERNEL(float);
REGISTER_BATCH_NORM_STATS_KERNEL(double);

}  // namespace oneflow
