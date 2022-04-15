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
#include "oneflow/user/kernels/multi_reduce_kernels.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/device/cuda_util.h"
#include <cub/cub.cuh>
#include <limits>

namespace oneflow {

namespace {

constexpr int64_t kMultiReduceMaxPackSize = 64;

template<typename T>
struct MultiReduceParamsPack {
  MultiReduceParam<T> params[kMultiReduceMaxPackSize];
  size_t size;
};

template<typename T, typename TransformFn, typename ReduceFn>
__global__ void MultiBlockReduceGpu(TransformFn transform,
                                    const MultiReduceParamsPack<T> pack_params, const T init,
                                    T* out) {
  ReduceFn reduce_fn{};
  T t_out = init;
  for (int i = 0; i < pack_params.size; ++i) {
    const auto& param = pack_params.params[i];
    CUDA_1D_KERNEL_LOOP(j, param.size) { t_out = reduce_fn(t_out, transform(param.data[j])); }
  }
  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T b_out = BlockReduce(temp_storage).Reduce(t_out, reduce_fn);
  if (threadIdx.x == 0) { out[blockIdx.x] = b_out; }
}

size_t InferTempStorageSize(user_op::InferContext* ctx) {
  auto input_size = ctx->input_size("x");
  if (input_size == 0) { return 0; }
  int64_t max_elem_cnt = 0;
  int64_t pack_size = 0;
  int32_t num_blocks = 0;
  for (size_t i = 0; i < input_size; ++i) {
    int64_t elem_cnt = ctx->InputShape("x", i).elem_cnt();
    max_elem_cnt = std::max(max_elem_cnt, elem_cnt);
    pack_size++;
    if (pack_size == kMultiReduceMaxPackSize || i == input_size - 1) {
      CHECK_LT(max_elem_cnt, std::numeric_limits<int32_t>::max());
      num_blocks += BlocksNum4ThreadsNum(static_cast<int32_t>(max_elem_cnt));
      max_elem_cnt = 0;
      pack_size = 0;
    }
  }
  CHECK_LT(num_blocks, kCudaThreadsNumPerBlock * kCudaThreadsNumPerBlock * kCudaThreadsNumPerBlock)
      << "Too much blocks needed for computing " << ctx->op_name() << ", should be less than "
      << kCudaThreadsNumPerBlock << "*" << kCudaThreadsNumPerBlock << "*" << kCudaThreadsNumPerBlock
      << ", but got " << num_blocks;
  size_t elem_size = GetSizeOfDataType(ctx->InputDType("x", 0));
  return GetCudaAlignedSize(num_blocks * elem_size * 2);
}

}  // namespace

template<typename T, typename TransformFn, typename ReduceFn>
struct MultiReduce<DeviceType::kCUDA, T, TransformFn, ReduceFn> {
  void operator()(ep::Stream* stream, TransformFn transform,
                  const std::vector<MultiReduceParam<T>>& params, T init, T* ret, T* temp) {
    CHECK_NOTNULL(temp);
    int32_t total_num_blocks = 0;
    for (size_t i = 0; i < params.size(); i += kMultiReduceMaxPackSize) {
      MultiReduceParamsPack<T> pack_params{};
      size_t max_elem_cnt = 0;
      pack_params.size = std::min<size_t>(kMultiReduceMaxPackSize, params.size() - i);
      for (size_t j = 0; j < pack_params.size; ++j) {
        pack_params.params[j] = params[i + j];
        max_elem_cnt = std::max<size_t>(max_elem_cnt, pack_params.params[j].size);
      }
      int32_t num_blocks = BlocksNum4ThreadsNum(max_elem_cnt);
      MultiBlockReduceGpu<T, TransformFn, ReduceFn>
          <<<num_blocks, kCudaThreadsNumPerBlock, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              transform, pack_params, init, temp + total_num_blocks);
      total_num_blocks += num_blocks;
    }
    size_t wksp_size = 0;
    auto DeviceReduce = [&](void* temp_storage) -> void {
      OF_CUDA_CHECK(cub::DeviceReduce::Reduce(temp_storage, wksp_size, temp, ret, total_num_blocks,
                                              ReduceFn{}, init,
                                              stream->As<ep::CudaStream>()->cuda_stream()));
    };
    DeviceReduce(nullptr);
    // NOTE(zwx): We have allocated the temp storage with the space
    //  that can hold all the elements to reduce,
    //  normally the `temp_storage_bytes` for cub::DeviceReduce shouldn't exceed it.
    CHECK_LE(wksp_size, total_num_blocks * sizeof(T))
        << wksp_size << " size in bytes of temp storage is needed for doing cub::DeviceReduce, "
        << "but only allocated " << total_num_blocks * sizeof(T);
    DeviceReduce(temp + total_num_blocks);
  }
};

#define REGISTER_MULTI_REDUCE_SUM_POW_ABS_CUDA_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("multi_reduce_sum_pow_abs")                                     \
      .SetCreateFn<MultiReduceSumPowAbsKernel<DeviceType::kCUDA, dtype>>()             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTempStorageSize);

#define REGISTER_MULTI_REDUCE_XIMUM_ABS_CUDA_KERNEL(op_type_name, ximum_enum, dtype)   \
  REGISTER_USER_KERNEL(op_type_name)                                                   \
      .SetCreateFn<MultiReduceXimumAbsKernel<DeviceType::kCUDA, dtype, ximum_enum>>()  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTempStorageSize);

#define REGISTER_MULTI_REDUCE_XIMUM_ABS_CUDA_KERNELS(dtype)                                     \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_CUDA_KERNEL("multi_reduce_max_abs", Ximum::kMax, dtype)       \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_CUDA_KERNEL("multi_reduce_min_abs", Ximum::kMin, dtype)       \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_CUDA_KERNEL("local_multi_reduce_max_abs", Ximum::kMax, dtype) \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_CUDA_KERNEL("local_multi_reduce_min_abs", Ximum::kMin, dtype)

REGISTER_MULTI_REDUCE_SUM_POW_ABS_CUDA_KERNEL(float)
REGISTER_MULTI_REDUCE_SUM_POW_ABS_CUDA_KERNEL(double)

REGISTER_MULTI_REDUCE_XIMUM_ABS_CUDA_KERNELS(float)
REGISTER_MULTI_REDUCE_XIMUM_ABS_CUDA_KERNELS(double)

}  // namespace oneflow
