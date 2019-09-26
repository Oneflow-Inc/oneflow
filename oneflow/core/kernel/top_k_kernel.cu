#include "oneflow/core/kernel/top_k_kernel.cuh"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

#define MAX_POWER 16

int32_t PowOf2Floor(int32_t val) {
  int32_t ret = 0;
  for (int32_t i = 0; i <= MAX_POWER; i++) {
    ret = std::pow(2, i);
    if (val < ret) {
      return ret / 2;
    } else if (val == ret) {
      return ret;
    }
  }
  return -1;
}

int32_t PowOf2Ceil(int32_t val) {
  int32_t ret = 0;
  for (int32_t i = 0; i <= MAX_POWER; i++) {
    ret = std::pow(2, i);
    if (val <= ret) { return ret; }
  }
  return -1;
}

template<typename T>
__global__ void HeapTopKKernel(const T* in_ptr, const int32_t instance_num,
                               const int32_t instance_size, const int32_t k,
                               const int32_t heap_size, const int32_t init_index,
                               const T init_value, int32_t* out_ptr) {
  extern __shared__ char smem[];
  auto* shared_entries = reinterpret_cast<Entry<T>*>(smem);

  const T* input = in_ptr + blockIdx.x * instance_size;
  auto heap = Heap<T>(shared_entries + threadIdx.x * heap_size, heap_size, init_index, init_value);
  // Divide elements to be sorted into disjoint sets (# of sets == # of heaps).
  // Each thread in the thread block manipulate one heap to select top heap_size entries from
  // corresponding set
  for (int32_t i = threadIdx.x; i < instance_size; i += blockDim.x) {
    auto entry = Entry<T>(i, input[i]);
    if (entry > heap[0]) { heap.ReplaceRoot(entry); }
  }

  __syncthreads();

  // Merge all heaps to a unified, sorted array
  bitonicSort(shared_entries, blockDim.x * heap_size,
              [](const Entry<T>& x, const Entry<T>& y) { return x > y; });
  // Write top_k elements in sorted array to output
  int32_t* output = out_ptr + blockIdx.x * k;
  for (int32_t i = threadIdx.x; i < k; i += blockDim.x) {
    output[i] = shared_entries[i].GetIndex();
  }
}

__global__ void RadixSortTopKInitializeKernel(int32_t* indices_ptr, int32_t instance_size) {
  for (int32_t i = threadIdx.x; i < instance_size; i += blockDim.x) {
    indices_ptr[blockIdx.x * instance_size + i] = i;
  }
}

__global__ void RadixSortTopKWriteToOutputKernel(const int32_t* sorted_indices_ptr,
                                                 int32_t instance_size, int32_t k,
                                                 int32_t* output_ptr) {
  for (int32_t i = threadIdx.x; i < k; i += blockDim.x) {
    output_ptr[blockIdx.x * k + i] = sorted_indices_ptr[blockIdx.x * instance_size + i];
  }
}

}  // namespace

template<typename T>
void GpuHeapSelectionTopK(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num,
                          int32_t instance_size, int32_t k, int32_t* out_ptr) {
  // Use as many heaps as possible (# of heaps == # of threads in thread block).
  // Limitation 1, max shared memory: 48KB
  // We also need heap_size * num_heap to be pow-of-2 which is necessary for bitonic sort
  // implemented in our system
  const int32_t heap_size = PowOf2Ceil(k);
  const int32_t heap_byte_size = heap_size * sizeof(Entry<T>);
  int32_t num_heap = PowOf2Floor(kCudaMaxSharedMemoryByteSize / heap_byte_size);
  CHECK_GT(num_heap, 0);
  // Limitation 2: # of threads in a thread block
  if (num_heap > kCudaThreadsNumPerBlock) { num_heap = kCudaThreadsNumPerBlock; }

  // Calculate shared memory size in thread block
  const int64_t smem_size = num_heap * heap_byte_size;
  CHECK_LE(smem_size, kCudaMaxSharedMemoryByteSize);

  HeapTopKKernel<T><<<instance_num, num_heap, smem_size, ctx->cuda_stream()>>>(
      in_ptr, instance_num, instance_size, k, heap_size, GetMaxVal<int32_t>(), GetMinVal<T>(),
      out_ptr);
}

#define INSTANTIATE_GPU_HEAP_SELECTION_TOP_K(T, type_proto)                                 \
  template void GpuHeapSelectionTopK<T>(DeviceCtx * ctx, const T* in, int32_t instance_num, \
                                        int32_t instance_size, int32_t k, int32_t* out_ptr);
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_HEAP_SELECTION_TOP_K, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_GPU_HEAP_SELECTION_TOP_K

template<typename T>
void GpuRadixSortTopK(DeviceCtx* ctx, const T* in_ptr, int32_t* indices_ptr, int32_t instance_num,
                      int32_t instance_size, int32_t k, void* temp_storage_ptr,
                      size_t temp_storage_bytes, T* sorted_in_ptr, int32_t* sorted_indices_ptr,
                      int32_t* out_ptr) {
  int32_t num_thread =
      instance_size <= kCudaThreadsNumPerBlock ? instance_size : kCudaThreadsNumPerBlock;
  RadixSortTopKInitializeKernel<<<instance_num, num_thread, 0, ctx->cuda_stream()>>>(indices_ptr,
                                                                                     instance_size);
  SortPairsDescending(in_ptr, indices_ptr, instance_num, instance_size, temp_storage_ptr,
                      temp_storage_bytes, sorted_in_ptr, sorted_indices_ptr, ctx->cuda_stream());
  num_thread = k <= kCudaThreadsNumPerBlock ? k : kCudaThreadsNumPerBlock;
  RadixSortTopKWriteToOutputKernel<<<instance_num, num_thread, 0, ctx->cuda_stream()>>>(
      sorted_indices_ptr, instance_size, k, out_ptr);
}

#define INSTANTIATE_GPU_RADIX_SORT_TOP_K(T, type_proto)                                    \
  template void GpuRadixSortTopK<T>(                                                       \
      DeviceCtx * ctx, const T* in_ptr, int32_t* indices_ptr, int32_t instance_num,        \
      int32_t instance_size, int32_t k, void* temp_storage_ptr, size_t temp_storage_bytes, \
      T* sorted_in_ptr, int32_t* sorted_indices_ptr, int32_t* out_ptr);
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_RADIX_SORT_TOP_K, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_GPU_RADIX_SORT_TOP_K

}  // namespace oneflow
