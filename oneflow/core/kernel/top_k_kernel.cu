#include "oneflow/core/kernel/top_k_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/top_k_heap_selection.cuh"
#include "oneflow/core/kernel/gpu_bitonic_sort.cuh"

#include <iostream>

namespace oneflow {

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
__global__ void HeapTopKKernel(const T* in, const int32_t instance_num, const int32_t instance_size,
                               const int32_t k, const int32_t heap_size, const int32_t init_index,
                               const T init_value, T* out) {
  extern __shared__ char smem[];
  auto* shared_entries = reinterpret_cast<Entry<T>*>(smem);

  const T* input = in + blockIdx.x * instance_size;
  auto heap = Heap<T>(shared_entries + threadIdx.x * heap_size, heap_size, init_index, init_value);
  // Divide elements to be sorted into disjoint sets (# of sets == # of heaps).
  // Each thread in the thread block manipulate one heap to select top heap_size entries from
  // corresponding set
  for (int32_t i = threadIdx.x; i < instance_size; i += blockDim.x) {
    auto entry = Entry<T>(i, input[i]);
    if (entry > heap[0]) { heap.ReplaceRoot(entry); }
  }
  // Merge all heaps to a unified, sorted array
  bitonicSort<Entry<T>, EntryGTComp<Entry<T>>>(shared_entries, blockDim.x * heap_size);
  T* output = out + blockIdx.x * (blockDim.x * heap_size);
  for (int32_t i = 0; i < blockDim.x * heap_size; ++i) {
    output[i] = static_cast<T>(shared_entries[i].GetIndex());
  }

  // Write top_k elements in sorted array to output
  // int32_t* output = out + blockIdx.x * k;
  // for (int32_t i = 0; i < k; ++i) { output[i] = shared_entries[i].GetIndex(); }
}

template<typename T>
struct TopKKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const T* in, const int32_t instance_num,
                      const int32_t instance_size, const int32_t k, const bool sorted,
                      int32_t* fw_buf, T* out) {
    CHECK(fw_buf == nullptr);
    // if (instance_size <= 1000 || k == instance_size || k > 512) {
    if (false) {
      TODO();
    } else {
      // Use as many heaps as possible (# of heaps == # of threads in thread block).
      // Limitation 1, max shared memory: 48KB
      // We also need heap_size * num_heap to be pow-of-2 which is necessary for bitonic sort
      // implemented in our system
      const int32_t heap_size = PowOf2Ceil(k);
      std::cout << "heap_size: " << heap_size << std::endl;
      const int32_t heap_byte_size = heap_size * sizeof(Entry<T>);
      std::cout << "heap_byte_size: " << heap_byte_size << std::endl;
      int32_t num_heap = PowOf2Floor(kCudaMaxSharedMemoryByteSize / heap_byte_size);
      CHECK_GT(num_heap, 0);
      // Limitation 2: # of threads in a thread block
      if (num_heap > kCudaThreadsNumPerBlock) { num_heap = kCudaThreadsNumPerBlock; }
      std::cout << "num_heap: " << num_heap << std::endl;

      // Calculate shared memory size in thread block
      const int64_t smem_size = num_heap * heap_byte_size;
      CHECK_LE(smem_size, kCudaMaxSharedMemoryByteSize);
      std::cout << "smem_size: " << smem_size << std::endl;

      std::cout << "max int32_t: " << GetMaxVal<int32_t>() << std::endl;
      std::cout << "min T: " << GetMinVal<T>() << std::endl;

      HeapTopKKernel<T><<<instance_num, num_heap, smem_size, ctx->cuda_stream()>>>(
          in, instance_num, instance_size, k, heap_size, GetMaxVal<int32_t>(), GetMinVal<T>(), out);
    }
  }
};

#define INSTANTIATE_TOP_K_KERNEL_UTIL(type_cpp, type_proto) \
  template struct TopKKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TOP_K_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_TOP_K_KERNEL_UTIL

}  // namespace oneflow
