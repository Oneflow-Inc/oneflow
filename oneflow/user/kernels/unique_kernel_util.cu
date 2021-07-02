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
#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include "oneflow/core/common/permutation_iterator.h"
#include "oneflow/core/common/not_equal_to_previous_adjacent_iterator.h"

namespace oneflow {

namespace {

template<typename T>
struct Buffer final {
  T* ptr = nullptr;
  size_t size_in_bytes = 0;
};

template<typename T>
int64_t GetTempBufferSize(int64_t n) {
  return GetCudaAlignedSize(n * sizeof(T));
}

template<typename KEY, typename IDX>
int64_t GetCubSortTempStorageSize(int64_t n) {
  size_t cub_sort_temp_store_size = 0;
  OF_CUDA_CHECK((cub::DeviceRadixSort::SortPairs<KEY, IDX>(nullptr, cub_sort_temp_store_size,
                                                           nullptr, nullptr, nullptr, nullptr, n)));
  CHECK_GE(cub_sort_temp_store_size, 0);
  CHECK_LT(cub_sort_temp_store_size, GetMaxVal<int64_t>());
  return GetCudaAlignedSize(static_cast<int64_t>(cub_sort_temp_store_size));
}

template<typename KEY, typename IDX>
int64_t GetCubScanTempStorageSize(int64_t n) {
  size_t cub_scan_temp_store_size = 0;
  NotEqualToPreviousAdjacentIterator<IDX, KEY> unique_counting_iter(nullptr, 0);
  PermutationIterator<IDX, IDX*, IDX*> remapping_iter(nullptr, nullptr);
  OF_CUDA_CHECK((cub::DeviceScan::InclusiveSum<NotEqualToPreviousAdjacentIterator<IDX, KEY>,
                                               PermutationIterator<IDX, IDX*, IDX*>>(
      nullptr, cub_scan_temp_store_size, unique_counting_iter, remapping_iter, n)));
  CHECK_GE(cub_scan_temp_store_size, 0);
  CHECK_LT(cub_scan_temp_store_size, GetMaxVal<int64_t>());
  return GetCudaAlignedSize(static_cast<int64_t>(cub_scan_temp_store_size));
}

template<typename KEY, typename IDX>
int64_t GetCubRleTempStorageSize(int64_t n) {
  size_t cub_rle_temp_store_size = 0;
  OF_CUDA_CHECK((cub::DeviceRunLengthEncode::Encode<KEY*, KEY*, IDX*, int64_t*>(
      nullptr, cub_rle_temp_store_size, nullptr, nullptr, nullptr, nullptr, n)));
  CHECK_GE(cub_rle_temp_store_size, 0);
  CHECK_LT(cub_rle_temp_store_size, GetMaxVal<int64_t>());
  return GetCudaAlignedSize(static_cast<int64_t>(cub_rle_temp_store_size));
}

template<typename KEY, typename IDX>
int64_t GetCubTempStorageSize(int64_t n) {
  int64_t cub_temp_storage_size = 0;
  cub_temp_storage_size = std::max(cub_temp_storage_size, GetCubSortTempStorageSize<KEY, IDX>(n));
  cub_temp_storage_size = std::max(cub_temp_storage_size, GetCubScanTempStorageSize<KEY, IDX>(n));
  cub_temp_storage_size = std::max(cub_temp_storage_size, GetCubRleTempStorageSize<KEY, IDX>(n));
  return cub_temp_storage_size;
}

template<typename T>
void AliasPtr(void* origin, int64_t* offset, Buffer<T>* buffer, int64_t size) {
  auto* ptr = reinterpret_cast<unsigned char*>(origin);
  if (buffer != nullptr) {
    buffer->ptr = reinterpret_cast<T*>(ptr + *offset);
    buffer->size_in_bytes = size;
  }
  *offset += size;
}

template<typename KEY, typename IDX>
void UniqueAliasWorkspace(DeviceCtx* ctx, int64_t n, void* workspace,
                          int64_t* workspace_size_in_bytes, Buffer<KEY>* cub_sort_keys_out,
                          Buffer<IDX>* cub_sort_values_out, Buffer<void>* cub_temp_storage) {
  int64_t offset = 0;
  AliasPtr(workspace, &offset, cub_sort_keys_out, GetTempBufferSize<KEY>(n));
  AliasPtr(workspace, &offset, cub_sort_values_out, GetTempBufferSize<IDX>(n));
  AliasPtr(workspace, &offset, cub_temp_storage, GetCubTempStorageSize<KEY, IDX>(n));
  *workspace_size_in_bytes = offset;
}

template<typename IDX>
__global__ void IotaKernel(int64_t n, IDX* out) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, n) { out[i] = static_cast<IDX>(i); }
}

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
  int64_t count_size = GetTempBufferSize<IDX>(n);
  UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::UniqueWithCounts(
      ctx, n, in, num_unique, unique_out, idx_out, reinterpret_cast<IDX*>(workspace),
      reinterpret_cast<unsigned char*>(workspace) + count_size,
      workspace_size_in_bytes - count_size);
}

template<typename KEY, typename IDX>
void UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::UniqueWithCounts(
    DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique, KEY* unique_out, IDX* idx_out,
    IDX* count, void* workspace, int64_t workspace_size_in_bytes) {
  int64_t rt_workspace_size;
  IDX* cub_sort_values_in_ptr = idx_out;
  Buffer<KEY> cub_sort_keys_out;
  Buffer<IDX> cub_sort_values_out;
  Buffer<void> cub_temp_storage;
  UniqueAliasWorkspace<KEY, IDX>(ctx, n, workspace, &rt_workspace_size, &cub_sort_keys_out,
                                 &cub_sort_values_out, &cub_temp_storage);
  CHECK_LE(rt_workspace_size, workspace_size_in_bytes);
  IotaKernel<IDX><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, cub_sort_values_in_ptr);
  OF_CUDA_CHECK((cub::DeviceRadixSort::SortPairs<KEY, IDX>(
      cub_temp_storage.ptr, cub_temp_storage.size_in_bytes, in, cub_sort_keys_out.ptr,
      cub_sort_values_in_ptr, cub_sort_values_out.ptr, n, 0, sizeof(KEY) * 8, ctx->cuda_stream())));
  OF_CUDA_CHECK((cub::DeviceRunLengthEncode::Encode<KEY*, KEY*, IDX*, IDX*>(
      cub_temp_storage.ptr, cub_temp_storage.size_in_bytes, cub_sort_keys_out.ptr, unique_out,
      count, num_unique, n, ctx->cuda_stream())));
  NotEqualToPreviousAdjacentIterator<IDX, KEY> unique_counting_iter(cub_sort_keys_out.ptr, 0);
  PermutationIterator<IDX, IDX*, IDX*> remapping_iter(idx_out, cub_sort_values_out.ptr);
  OF_CUDA_CHECK((cub::DeviceScan::InclusiveSum<NotEqualToPreviousAdjacentIterator<IDX, KEY>,
                                               PermutationIterator<IDX, IDX*, IDX*>>(
      cub_temp_storage.ptr, cub_temp_storage.size_in_bytes, unique_counting_iter, remapping_iter, n,
      ctx->cuda_stream())));
}

template<typename KEY, typename IDX>
void UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::GetUniqueWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes) {
  UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::GetUniqueWithCountsWorkspaceSizeInBytes(
      ctx, n, workspace_size_in_bytes);
  *workspace_size_in_bytes += GetTempBufferSize<IDX>(n);
}

template<typename KEY, typename IDX>
void UniqueKernelUtil<DeviceType::kGPU, KEY, IDX>::GetUniqueWithCountsWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t* workspace_size_in_bytes) {
  UniqueAliasWorkspace<KEY, IDX>(ctx, n, nullptr, workspace_size_in_bytes, nullptr, nullptr,
                                 nullptr);
}

#define INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU(key_type_pair, idx_type_pair)              \
  template struct UniqueKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(key_type_pair), \
                                   OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU, ARITHMETIC_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_UNIQUE_KERNEL_UTIL_GPU

}  // namespace oneflow
