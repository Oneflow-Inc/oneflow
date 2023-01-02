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
// The implementation of batch_norm_backward_elemt kernel is modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Normalization.cuh

namespace oneflow {

namespace {

template<typename T, typename ACC_T, typename IDX_TYPE>
__global__ void batch_norm_backward_elemt_kernel(
    const IDX_TYPE batch_size, const IDX_TYPE channel_size, const IDX_TYPE spatial_size,
    const T* grad_out_ptr, const T* input_ptr, const T* mean_ptr, const T* invstd_ptr,
    const T* weight_ptr, const T* sum_dy_ptr, const T* sum_dy_xmu_ptr, T* grad_in_ptr,
    const int32_t* count_ptr, const int64_t world_size) {
  int64_t total_numel = 0;
  for (int i = 0; i < world_size; i++) { total_numel += count_ptr[i]; }

  const ACC_T norm_fct = static_cast<ACC_T>(1) / static_cast<ACC_T>(total_numel);

  IDX_TYPE channel = blockIdx.x;

  if (channel >= channel_size) { return; }

  ACC_T m_c = mean_ptr[channel];
  ACC_T m_dy_c = sum_dy_ptr[channel] * norm_fct;
  ACC_T factor_1_c = invstd_ptr[channel];
  ACC_T factor_2_c = static_cast<ACC_T>(weight_ptr[channel]);
  factor_2_c *= factor_1_c;
  factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu_ptr[channel] * norm_fct;

  IDX_TYPE batch_offset = spatial_size * channel_size;
  IDX_TYPE channel_offset = channel * spatial_size;

  IDX_TYPE bstep = blockDim.y * gridDim.y;
  for (IDX_TYPE batch = threadIdx.y + blockIdx.y * blockDim.y; batch < batch_size; batch += bstep) {
    IDX_TYPE offset = batch * batch_offset;
    for (IDX_TYPE feature = threadIdx.x; feature < spatial_size; feature += blockDim.x) {
      grad_in_ptr[offset + channel_offset + feature] =
          static_cast<T>((grad_out_ptr[offset + channel_offset + feature] - m_dy_c
                          - (input_ptr[offset + channel_offset + feature] - m_c) * factor_1_c)
                         * factor_2_c);
    }
  }
}

template<typename T, typename ACC_T, typename IDX_TYPE, int PARALLEL_LOADS>
__global__ void batch_norm_backward_elemt_channels_last_kernel(
    const T* grad_out_ptr, const T* input_ptr, const ACC_T* mean_ptr, const ACC_T* invstd_ptr,
    const T* weight_ptr, const ACC_T* sum_dy_ptr, const ACC_T* sum_dy_xmu_ptr,
    const int32_t* count_ptr, T* grad_in_ptr, const IDX_TYPE world_size, const IDX_TYPE stride,
    const IDX_TYPE reduction_size) {
  IDX_TYPE total_numel = 0;
  for (IDX_TYPE i = 0; i < world_size; i++) { total_numel += count_ptr[i]; }

  auto norm_fct = static_cast<ACC_T>(1) / static_cast<ACC_T>(total_numel);

  // tensor dimension (m,c)
  // loop along m dimension
  IDX_TYPE inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  IDX_TYPE m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  IDX_TYPE c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (c_offset >= stride || m_offset >= reduction_size) { return; }

  auto m_c = mean_ptr[c_offset];
  auto m_dy_c = sum_dy_ptr[c_offset] * norm_fct;
  auto factor_1_c = invstd_ptr[c_offset];
  auto factor_2_c =
      (weight_ptr == nullptr ? ACC_T(1.0) : static_cast<ACC_T>(weight_ptr[c_offset])) * factor_1_c;
  factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu_ptr[c_offset] * norm_fct;

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        grad_in_ptr[address_base] =
            static_cast<T>((static_cast<ACC_T>(grad_out_ptr[address_base]) - m_dy_c
                            - (static_cast<ACC_T>(input_ptr[address_base]) - m_c) * factor_1_c)
                           * factor_2_c);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }
  }
}

template<typename T>
struct BatchNormBackwardElemtFunctor final {
  void operator()(ep::Stream* stream, const int64_t batch_size, const int64_t channel_size,
                  const int64_t spatial_size, const T* grad_out_ptr, const T* input_ptr,
                  const T* mean_ptr, const T* invstd_ptr, const T* weight_ptr, const T* sum_dy_ptr,
                  const T* sum_dy_xmu_ptr, T* grad_in_ptr, const int32_t* count_ptr,
                  const int64_t world_size) {
    using ACC_T = acc_type<T>;

    // The kernel is pointwise, but we need to balance reading parameters (save_var/mean,
    // weight/bias) - which we only do once and have a for loop afterwards - with having many
    // threads and blocks and good occupancy. Quiet likely, we could go with even more blocks than
    // 1024. The various planes are independent, so we use blocks for them.
    int tf = std::max<int>(getNumThreads(spatial_size / 4),
                           std::min<int>(getNumThreads(spatial_size), 64));
    int tb = std::max<int>(64 / tf, 1);
    dim3 blocks_trans(channel_size, std::max<int>(1, std::min<int>((256 * 1024) / channel_size,
                                                                   (batch_size + tb - 1) / tb)));
    blocks_trans.y = std::min(blocks_trans.y, MAX_GRID_SIZE);
    dim3 threads_trans(tf, tb);

    if (batch_size * channel_size * spatial_size < std::numeric_limits<int32_t>::max()) {
      batch_norm_backward_elemt_kernel<T, ACC_T, int32_t>
          <<<blocks_trans, threads_trans, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              static_cast<int32_t>(batch_size), static_cast<int32_t>(channel_size),
              static_cast<int32_t>(spatial_size), grad_out_ptr, input_ptr, mean_ptr, invstd_ptr,
              weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr, grad_in_ptr, count_ptr, world_size);
    } else {
      batch_norm_backward_elemt_kernel<T, ACC_T, int64_t>
          <<<blocks_trans, threads_trans, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              batch_size, channel_size, spatial_size, grad_out_ptr, input_ptr, mean_ptr, invstd_ptr,
              weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr, grad_in_ptr, count_ptr, world_size);
    }
  }
};

template<typename T>
struct BatchNormBackwardElemtChannelLastFunctor final {
  void operator()(ep::Stream* stream, const int64_t stride, const int64_t reduction_size,
                  const T* grad_out_ptr, const T* input_ptr, const T* mean_ptr, const T* invstd_ptr,
                  const T* weight_ptr, const T* sum_dy_ptr, const T* sum_dy_xmu_ptr, T* grad_in_ptr,
                  const int32_t* count_ptr, const int64_t world_size) {
    using ACC_T = acc_type<T>;
    dim3 block;
    dim3 grid;
    flexible_launch_configs(reduction_size, stride, block, grid);

    if (stride * reduction_size < std::numeric_limits<int32_t>::max()) {
      batch_norm_backward_elemt_channels_last_kernel<T, ACC_T, int32_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              grad_out_ptr, input_ptr, mean_ptr, invstd_ptr, weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr,
              count_ptr, grad_in_ptr, world_size, static_cast<int32_t>(stride),
              static_cast<int32_t>(reduction_size));
    } else {
      batch_norm_backward_elemt_channels_last_kernel<T, ACC_T, int64_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              grad_out_ptr, input_ptr, mean_ptr, invstd_ptr, weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr,
              count_ptr, grad_in_ptr, world_size, stride, reduction_size);
    }
  }
};

}  // namespace

template<typename T>
class GpuBatchNormBackwardElemtKernel final : public user_op::OpKernel {
 public:
  GpuBatchNormBackwardElemtKernel() = default;
  ~GpuBatchNormBackwardElemtKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* grad_out = ctx->Tensor4ArgNameAndIndex("grad_out", 0);
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* invstd = ctx->Tensor4ArgNameAndIndex("invstd", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* sum_dy = ctx->Tensor4ArgNameAndIndex("sum_dy", 0);
    const user_op::Tensor* sum_dy_xmu = ctx->Tensor4ArgNameAndIndex("sum_dy_xmu", 0);
    const user_op::Tensor* count = ctx->Tensor4ArgNameAndIndex("count", 0);

    user_op::Tensor* grad_in = ctx->Tensor4ArgNameAndIndex("grad_in", 0);

    const T* grad_out_ptr = grad_out->dptr<T>();
    const T* input_ptr = input->dptr<T>();
    const T* mean_ptr = mean->dptr<T>();
    const T* invstd_ptr = invstd->dptr<T>();
    const T* weight_ptr = weight->dptr<T>();
    const T* sum_dy_ptr = sum_dy->dptr<T>();
    const T* sum_dy_xmu_ptr = sum_dy_xmu->dptr<T>();
    const int32_t* count_ptr = count->dptr<int32_t>();

    T* grad_in_ptr = grad_in->mut_dptr<T>();
    const int32_t axis = ctx->Attr<int32_t>("axis");

    bool use_channels_last_kernel = axis == 1 ? false : true;
    const int64_t world_size = count->shape_view().elem_cnt();
    if (use_channels_last_kernel) {  // NHWC format
      const int64_t stride = input->shape_view().At(axis);
      const int64_t reduction_size = input->shape_view().elem_cnt() / stride;
      BatchNormBackwardElemtChannelLastFunctor<T>()(
          ctx->stream(), stride, reduction_size, grad_out_ptr, input_ptr, mean_ptr, invstd_ptr,
          weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr, grad_in_ptr, count_ptr, world_size);
    } else {  // NCHW format
      const int64_t batch_size = input->shape_view().At(0);
      const int64_t channel_size = input->shape_view().At(1);
      const int64_t spatial_size = input->shape_view().Count(2);

      BatchNormBackwardElemtFunctor<T>()(
          ctx->stream(), batch_size, channel_size, spatial_size, grad_out_ptr, input_ptr, mean_ptr,
          invstd_ptr, weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr, grad_in_ptr, count_ptr, world_size);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BATCH_NORM_BACKWARD_ELEMT_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("batch_norm_backward_elemt")                                            \
      .SetCreateFn<GpuBatchNormBackwardElemtKernel<dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                         \
                       && (user_op::HobDataType("grad_out", 0) == GetDataType<dtype>::value)   \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)      \
                       && (user_op::HobDataType("mean", 0) == GetDataType<dtype>::value)       \
                       && (user_op::HobDataType("invstd", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("weight", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("sum_dy", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("sum_dy_xmu", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("count", 0) == GetDataType<int32_t>::value))

REGISTER_BATCH_NORM_BACKWARD_ELEMT_KERNEL(float);
REGISTER_BATCH_NORM_BACKWARD_ELEMT_KERNEL(double);

}  // namespace oneflow
