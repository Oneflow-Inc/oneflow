#ifndef GPU_BITONIC_SORT_CUH
#define GPU_BITONIC_SORT_CUH

#include <assert.h>

template<typename T>
__device__ const bool IsIntegerPowerOf2(const T v) {
  return (v > 0 && !(v & (v - 1)));
}

template<typename T, typename Comparator>
__device__ void bitonicSwap(T* x, T* y, const bool dir) {
  Comparator comp = Comparator();
  if (comp(*x, *y) == dir) {
    T tmp = *x;
    *x = *y;
    *y = tmp;
  }
}

template<typename T, typename Comparator>
__device__ void bitonicSort(T* data, const int32_t elem_cnt) {
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

        bitonicSwap<T, Comparator>(data + pos, data + pos + stride, dir);

        __syncthreads();
      }
    }
  }

  // Sort based on the bitonic sequence
  for (int32_t stride = elem_cnt / 2; stride > 0; stride /= 2) {
    for (int32_t swap_id = threadIdx.x; swap_id < elem_cnt / 2; swap_id += blockDim.x) {
      // Locate the pair {pos, pos + stride} which is going te be swaped if needed
      const int pos = 2 * swap_id - (swap_id & (stride - 1));

      bitonicSwap<T, Comparator>(data + pos, data + pos + stride, false);

      __syncthreads();
    }
  }
}

#endif
