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
#include "oneflow/user/kernels/unique_kernel_util.h"
#include "oneflow/core/cuda/unique.cuh"

namespace oneflow {

namespace {

constexpr cuda::unique::Flag kUniqueFlag = cuda::unique::kOutputInverseIndices;
constexpr cuda::unique::Flag kUniqueWithCountsFlag =
    cuda::unique::kOutputInverseIndices | cuda::unique::kOutputCounts;

}  // namespace

template<typename KEY, typename IDX>
struct UniqueKernelUtil<DeviceType::kGPU, KEY, IDX> {
  static void Unique(DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique, KEY* unique_out,
                     IDX* idx_out, void* workspace, int64_t workspace_size_in_bytes);
  static void UniqueWithCounts(DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique,
                               KEY* unique_out, IDX* idx_out, IDX* count, void* workspace,
                               int64_t workspace_size_in_bytes);
  static void GetUniqueWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                            int64_t* workspace_size_in_bytes);
  static void GetUniqueWithCountsWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                                      int64_t* workspace_size_in_bytes);
};

template<typename KEY, typename IDX>
void UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::Unique(DeviceCtx* ctx, int64_t n, const KEY* in,
                                                          IDX* num_unique, KEY* unique_out,
                                                          IDX* idx_out, void* workspace,
                                                          int64_t workspace_size_in_bytes) {
  OF_CUDA_CHECK(
      (cuda::unique::Launch<KEY, IDX>(kUniqueFlag, n, in, unique_out, num_unique, idx_out, nullptr,
                                      workspace, workspace_size_in_bytes, ctx->cuda_stream())));
}

template<typename KEY, typename IDX>
void UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::UniqueWithCounts(
    DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique, KEY* unique_out, IDX* idx_out,
    IDX* count, void* workspace, int64_t workspace_size_in_bytes) {
  OF_CUDA_CHECK((cuda::unique::Launch<KEY, IDX>(kUniqueWithCountsFlag, n, in, unique_out,
                                                num_unique, idx_out, count, workspace,
                                                workspace_size_in_bytes, ctx->cuda_stream())));
}

template<typename KEY, typename IDX>
void UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::GetUniqueWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes) {
  size_t ws = 0;
  OF_CUDA_CHECK((cuda::unique::GetWorkspaceSize<KEY, IDX>(kUniqueFlag, n, &ws)));
  *workspace_size_in_bytes = static_cast<int64_t>(ws);
}

template<typename KEY, typename IDX>
void UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::GetUniqueWithCountsWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes) {
  size_t ws = 0;
  OF_CUDA_CHECK((cuda::unique::GetWorkspaceSize<KEY, IDX>(kUniqueWithCountsFlag, n, &ws)));
  *workspace_size_in_bytes = static_cast<int64_t>(ws);
}

#define INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU(key_type_pair, idx_type_pair)              \
  template struct UniqueKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(key_type_pair), \
                                   OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU, ARITHMETIC_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU

}  // namespace oneflow
