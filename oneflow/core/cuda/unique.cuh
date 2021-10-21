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
#ifndef ONEFLOW_CORE_CUDA_UNIQUE_H_
#define ONEFLOW_CORE_CUDA_UNIQUE_H_

#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include "oneflow/core/common/permutation_iterator.h"
#include "oneflow/core/common/not_equal_to_previous_adjacent_iterator.h"

namespace oneflow {

namespace cuda {

namespace unique {

using Flag = uint32_t;
static constexpr Flag kDefault = 0x0;
static constexpr Flag kInputSorted = 0x1;
static constexpr Flag kOutputInverseIndices = 0x1 << 1;
static constexpr Flag kOutputCounts = 0x1 << 2;

namespace {

constexpr size_t kCudaAlignSize = 512;

__device__ __host__ __forceinline__ size_t GetCudaAlignedSize(size_t size) {
  return (size + kCudaAlignSize - 1) / kCudaAlignSize * kCudaAlignSize;
}

template<typename T>
__device__ __host__ __forceinline__ T* PtrOffset(void* ptr, size_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(ptr) + offset);
}

__device__ __host__ __forceinline__ size_t max(size_t a, size_t b) { return a > b ? a : b; }

template<typename Key, typename Index>
cudaError_t DoUnique(size_t n, const Key* sorted_in, Key* unique, Index* num_unique,
                     void* workspace, size_t* workspace_size, cudaStream_t stream) {
  size_t ws = *workspace_size;
  cudaError_t err = cub::DeviceSelect::Unique<const Key*, Key*, Index*>(
      workspace, ws, sorted_in, unique, num_unique, n, stream);
  if (err != cudaSuccess) { return err; }
  if (*workspace_size == 0) { *workspace_size = ws; }
  return cudaSuccess;
}

template<typename Key, typename Index>
cudaError_t DoUniqueWithCounts(size_t n, const Key* sorted_in, Key* unique, Index* num_unique,
                               Index* counts, void* workspace, size_t* workspace_size,
                               cudaStream_t stream) {
  size_t ws = *workspace_size;
  cudaError_t err = cub::DeviceRunLengthEncode::Encode<const Key*, Key*, Index*, Index*>(
      workspace, ws, sorted_in, unique, counts, num_unique, n, stream);
  if (err != cudaSuccess) { return err; }
  if (*workspace_size == 0) { *workspace_size = ws; }
  return cudaSuccess;
}

template<typename Key, typename Index>
cudaError_t DispatchOutputCounts(Flag flag, size_t n, const Key* sorted_in, Key* unique,
                                 Index* num_unique, Index* counts, void* workspace,
                                 size_t* workspace_size, cudaStream_t stream) {
  size_t ws = *workspace_size;
  if ((flag & kOutputCounts) != 0) {
    cudaError_t err = DoUniqueWithCounts<Key, Index>(n, sorted_in, unique, num_unique, counts,
                                                     workspace, &ws, stream);
    if (err != cudaSuccess) { return err; }
  } else {
    cudaError_t err =
        DoUnique<Key, Index>(n, sorted_in, unique, num_unique, workspace, &ws, stream);
    if (err != cudaSuccess) { return err; }
  }
  if (*workspace_size == 0) { *workspace_size = ws; }
  return cudaSuccess;
}

template<typename Key, typename Index, typename InverseIndicesIter>
cudaError_t DoGenInverseIndices(size_t n, const Key* sorted_in,
                                InverseIndicesIter inverse_indices_iter, void* workspace,
                                size_t* workspace_size, cudaStream_t stream) {
  size_t ws = *workspace_size;
  NotEqualToPreviousAdjacentIterator<Index, Key> unique_counting_iter(sorted_in, 0);
  cudaError_t err =
      cub::DeviceScan::InclusiveSum<decltype(unique_counting_iter), InverseIndicesIter>(
          workspace, ws, unique_counting_iter, inverse_indices_iter, n, stream);
  if (err != cudaSuccess) { return err; }
  if (*workspace_size == 0) { *workspace_size = ws; }
  return cudaSuccess;
}

template<typename Key, typename Index, typename InverseIndicesIter>
cudaError_t DispatchOutputInverseIndices(Flag flag, size_t n, const Key* sorted_in, Key* unique,
                                         Index* num_unique, InverseIndicesIter inverse_indices_iter,
                                         Index* counts, void* workspace, size_t* workspace_size,
                                         cudaStream_t stream) {
  size_t dispatch_with_counts_ws = *workspace_size;
  size_t do_gen_inverse_indices_ws = *workspace_size;
  {
    cudaError_t err =
        DispatchOutputCounts<Key, Index>(flag, n, sorted_in, unique, num_unique, counts, workspace,
                                         &dispatch_with_counts_ws, stream);
    if (err != cudaSuccess) { return err; }
  }
  if ((flag & kOutputInverseIndices) != 0) {
    cudaError_t err = DoGenInverseIndices<Key, Index, InverseIndicesIter>(
        n, sorted_in, inverse_indices_iter, workspace, &do_gen_inverse_indices_ws, stream);
    if (err != cudaSuccess) { return err; }
  }
  if (*workspace_size == 0) {
    *workspace_size = max(dispatch_with_counts_ws, do_gen_inverse_indices_ws);
  }
  return cudaSuccess;
}

template<typename T>
__global__ void IotaKernel(size_t n, T* out) {
  for (T i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < n;
       i += step) {
    out[i] = i;
  }
}

template<typename Key, typename Index>
cudaError_t DoSort(size_t n, const Key* in, Key* sorted, Index* sorted_indices, void* workspace,
                   size_t* workspace_size, cudaStream_t stream) {
  Index* indices;
  const size_t indices_size = GetCudaAlignedSize(n * sizeof(Index));
  void* sort_workspace;
  size_t sort_ws;
  if (*workspace_size == 0) {
    indices = nullptr;
    sort_workspace = nullptr;
    sort_ws = 0;
  } else {
    if (*workspace_size <= indices_size) { return cudaErrorInvalidValue; }
    indices = PtrOffset<Index>(workspace, 0);
    sort_workspace = PtrOffset<Index>(workspace, indices_size);
    sort_ws = *workspace_size - indices_size;
  }
  if (*workspace_size != 0) {
    const int block_size = 1024;
    const int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    IotaKernel<Index><<<num_blocks, block_size, 0, stream>>>(n, indices);
  }
  cudaError_t err = cub::DeviceRadixSort::SortPairs<Key, Index>(
      sort_workspace, sort_ws, in, sorted, indices, sorted_indices, n, 0, sizeof(Key) * 8, stream);
  if (err != cudaSuccess) { return err; }
  if (*workspace_size == 0) { *workspace_size = indices_size + sort_ws; }
  return cudaSuccess;
}

template<typename Key, typename Index>
cudaError_t DispatchInputSorted(Flag flag, size_t n, const Key* in, Key* unique, Index* num_unique,
                                Index* inverse_indices, Index* counts, void* workspace,
                                size_t* workspace_size, cudaStream_t stream) {
  if ((flag & kInputSorted) != 0) {
    return DispatchOutputInverseIndices<Key, Index, Index*>(flag, n, in, unique, num_unique,
                                                            inverse_indices, counts, workspace,
                                                            workspace_size, stream);
  } else {
    const size_t sorted_in_size = GetCudaAlignedSize(n * sizeof(Key));
    const size_t sorted_indices_size = GetCudaAlignedSize(n * sizeof(Index));
    const size_t sort_buffer_size = sorted_in_size + sorted_indices_size;
    Key* sorted_in;
    Index* sorted_indices;
    size_t do_sort_ws;
    void* do_sort_workspace;
    size_t do_inverse_indices_ws;
    void* do_inverse_indices_workspace;
    if (*workspace_size == 0) {
      sorted_in = nullptr;
      sorted_indices = nullptr;
      do_sort_ws = 0;
      do_sort_workspace = nullptr;
      do_inverse_indices_ws = 0;
      do_inverse_indices_workspace = nullptr;
    } else {
      if (*workspace_size <= sort_buffer_size) { return cudaErrorInvalidValue; }
      sorted_in = PtrOffset<Key>(workspace, 0);
      sorted_indices = PtrOffset<Index>(workspace, sorted_in_size);
      do_sort_ws = *workspace_size - sort_buffer_size;
      do_sort_workspace = PtrOffset<void>(workspace, sort_buffer_size);
      do_inverse_indices_ws = do_sort_ws;
      do_inverse_indices_workspace = do_sort_workspace;
    }
    {
      cudaError_t err = DoSort<Key, Index>(n, in, sorted_in, sorted_indices, do_sort_workspace,
                                           &do_sort_ws, stream);
      if (err != cudaSuccess) { return err; }
    }
    PermutationIterator<Index, Index*, Index*> inverse_indices_iter(inverse_indices,
                                                                    sorted_indices);
    {
      cudaError_t err = DispatchOutputInverseIndices<Key, Index, decltype(inverse_indices_iter)>(
          flag, n, sorted_in, unique, num_unique, inverse_indices_iter, counts,
          do_inverse_indices_workspace, &do_inverse_indices_ws, stream);
      if (err != cudaSuccess) { return err; }
    }
    if (*workspace_size == 0) {
      *workspace_size = sort_buffer_size + max(do_sort_ws, do_inverse_indices_ws);
    }
    return cudaSuccess;
  }
}

}  // namespace

template<typename Key, typename Index>
cudaError_t Launch(Flag flag, size_t n, const Key* in, Key* unique, Index* num_unique,
                   Index* inverse_indices, Index* counts, void* workspace, size_t workspace_size,
                   cudaStream_t stream) {
  if (workspace_size == 0) { return cudaErrorInvalidValue; }
  return DispatchInputSorted<Key, Index>(flag, n, in, unique, num_unique, inverse_indices, counts,
                                         workspace, &workspace_size, stream);
}

template<typename Key, typename Index>
cudaError_t GetWorkspaceSize(Flag flag, size_t n, size_t* workspace_size) {
  *workspace_size = 0;
  return DispatchInputSorted<Key, Index>(flag, n, nullptr, nullptr, nullptr, nullptr, nullptr,
                                         nullptr, workspace_size, 0);
}

}  // namespace unique

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_UNIQUE_H_
