#ifndef GPU_BITONIC_SORT_CUH
#define GPU_BITONIC_SORT_CUH

#include <assert.h>

template<typename T>
__device__ const bool IsIntegerPowerOf2(const T v) {
  return (v > 0 && !(v & (v - 1)));
}

template<typename T, typename Compare>
__device__ void bitonicSwap(T* data, const int32_t i, const int32_t j, const bool dir,
                            const Compare& comp) {
  if (comp(data[i], data[j]) == dir) {
    T tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }
}

template<typename T, typename Compare>
__device__ void bitonicSort(T* data, const int32_t elem_cnt, const Compare& comp) {
  assert(IsIntegerPowerOf2(elem_cnt));
  assert(IsIntegerPowerOf2(blockDim.x));

  // Generate a bitonic sequence from input
  for (int32_t size = 2; size <= elem_cnt / 2; size *= 2) {
    // Merge 2 bitonic sequences of length 'size' into a bitonic sequence of length '2 * size'
    for (int32_t stride = size / 2; stride > 0; stride /= 2) {
      for (int32_t swap_id = threadIdx.x; swap_id < elem_cnt / 2; swap_id += blockDim.x) {
        // Change dir at intervals of 'size / 2' swaps
        const bool dir = swap_id & (size / 2);
        // Locate the pair {pos, pos + stride} which is going te be swaped if needed
        const int pos = 2 * swap_id - (swap_id & (stride - 1));

        bitonicSwap(data, pos, pos + stride, dir, comp);

        __syncthreads();
      }
    }
  }

  // Sort based on the bitonic sequence
  for (int32_t stride = elem_cnt / 2; stride > 0; stride /= 2) {
    for (int32_t swap_id = threadIdx.x; swap_id < elem_cnt / 2; swap_id += blockDim.x) {
      // Locate the pair {pos, pos + stride} which is going te be swaped if needed
      const int pos = 2 * swap_id - (swap_id & (stride - 1));

      bitonicSwap(data, pos, pos + stride, false, comp);

      __syncthreads();
    }
  }
}

#endif
