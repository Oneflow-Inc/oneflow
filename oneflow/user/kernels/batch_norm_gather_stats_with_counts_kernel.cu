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
// The implementation of batch_norm_gather_stats_with_counts kernel is modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Normalization.cuh

namespace oneflow {

namespace {

template<typename T, typename ACC_T, typename IDX_TYPE>
__global__ void batch_norm_reduce_statistics_kernel(const int64_t world_size,
                                                    const int64_t channel_size, const T* mean_ptr,
                                                    const T* invstd_ptr, const T* counts_ptr,
                                                    T* global_mean_ptr, T* global_invstd_ptr,
                                                    T* running_mean_ptr, T* running_var_ptr,
                                                    const float eps, const float momentum) {
  IDX_TYPE bid = blockIdx.x;
  IDX_TYPE tid = threadIdx.x;

  // first the reductions each thread does separately
  for (IDX_TYPE i = bid * blockDim.x + tid; i < channel_size; i += gridDim.x * blockDim.x) {
    ACC_T avg = 0;
    ACC_T var_n = 0;
    IDX_TYPE n = 0;
    for (IDX_TYPE j = 0; j < world_size; j++) {
      T count = counts_ptr[j];
      ACC_T m = mean_ptr[j * channel_size + i];
      ACC_T v = ACC_T(1.0) / (invstd_ptr[j * channel_size + i]);
      v = (v * v - eps) * count;
      ACC_T factor = 1.0 / (n + count);
      var_n += v + (avg - m) * (avg - m) * n * count * factor;
      avg = n * factor * avg + count * factor * m;
      n += count;
    }
    global_mean_ptr[i] = avg;
    global_invstd_ptr[i] = static_cast<ACC_T>(1) / device_sqrt(var_n / n + eps);
    if (running_mean_ptr != nullptr) {
      running_mean_ptr[i] = static_cast<T>((1 - momentum) * running_mean_ptr[i] + momentum * avg);
    }
    ACC_T unbiasedVar = var_n / (n - 1);
    if (running_var_ptr != nullptr) {
      running_var_ptr[i] =
          static_cast<T>((1 - momentum) * running_var_ptr[i] + momentum * unbiasedVar);
    }
  }
}

template<typename T>
struct BatchNormGatherStatsWithCountsFunctor final {
  void operator()(ep::Stream* stream, const int64_t world_size, const int64_t channel_size,
                  const T* mean_ptr, const T* invstd_ptr, const T* counts_ptr, T* global_mean_ptr,
                  T* global_invstd_ptr, T* running_mean_ptr, T* running_var_ptr, const float eps,
                  const float momentum) {
    using ACC_T = acc_type<T>;
    int32_t block = getNumThreads(channel_size);
    int32_t grid = std::max<int32_t>(1, channel_size / block);

    if (world_size * channel_size < std::numeric_limits<int32_t>::max()) {
      batch_norm_reduce_statistics_kernel<T, ACC_T, int32_t>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              static_cast<int32_t>(world_size), static_cast<int32_t>(channel_size), mean_ptr,
              invstd_ptr, counts_ptr, global_mean_ptr, global_invstd_ptr, running_mean_ptr,
              running_var_ptr, eps, momentum);
    } else {
      batch_norm_reduce_statistics_kernel<T, ACC_T, int64_t>
          <<<grid, block, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              world_size, channel_size, mean_ptr, invstd_ptr, counts_ptr, global_mean_ptr,
              global_invstd_ptr, running_mean_ptr, running_var_ptr, eps, momentum);
    }
  }
};

}  // namespace

template<typename T>
class GpuBatchNormGatherStatsWithCountsKernel final : public user_op::OpKernel {
 public:
  GpuBatchNormGatherStatsWithCountsKernel() = default;
  ~GpuBatchNormGatherStatsWithCountsKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* invstd = ctx->Tensor4ArgNameAndIndex("invstd", 0);
    const user_op::Tensor* counts = ctx->Tensor4ArgNameAndIndex("counts", 0);
    user_op::Tensor* global_mean = ctx->Tensor4ArgNameAndIndex("global_mean", 0);
    user_op::Tensor* global_invstd = ctx->Tensor4ArgNameAndIndex("global_invstd", 0);

    const T* mean_ptr = mean->dptr<T>();
    const T* invstd_ptr = invstd->dptr<T>();
    const T* counts_ptr = counts->dptr<T>();
    T* global_mean_ptr = global_mean->mut_dptr<T>();
    T* global_invstd_ptr = global_invstd->mut_dptr<T>();
    T* running_mean_ptr = nullptr;
    T* running_var_ptr = nullptr;
    if (ctx->has_input("running_mean", 0)) {
      CHECK(ctx->has_input("running_var", 0));
      running_mean_ptr = ctx->Tensor4ArgNameAndIndex("running_mean", 0)->mut_dptr<T>();
      running_var_ptr = ctx->Tensor4ArgNameAndIndex("running_var", 0)->mut_dptr<T>();
    }

    const float eps = ctx->Attr<float>("eps");
    const float momentum = ctx->Attr<float>("momentum");

    const int64_t world_size = mean->shape_view().At(0);
    const int64_t channel_size = mean->shape_view().At(1);

    BatchNormGatherStatsWithCountsFunctor<T>()(
        ctx->stream(), world_size, channel_size, mean_ptr, invstd_ptr, counts_ptr, global_mean_ptr,
        global_invstd_ptr, running_mean_ptr, running_var_ptr, eps, momentum);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BATCH_NORM_GATHER_STATS_WITH_COUNTS_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("batch_norm_gather_stats_with_counts")                              \
      .SetCreateFn<GpuBatchNormGatherStatsWithCountsKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("mean", 0) == GetDataType<dtype>::value)   \
                       && (user_op::HobDataType("invstd", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("counts", 0) == GetDataType<dtype>::value))

REGISTER_BATCH_NORM_GATHER_STATS_WITH_COUNTS_KERNEL(float);
REGISTER_BATCH_NORM_GATHER_STATS_WITH_COUNTS_KERNEL(double);

}  // namespace oneflow
