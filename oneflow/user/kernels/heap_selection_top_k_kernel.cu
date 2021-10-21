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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T>
T PowOf2Floor(T val, int32_t max_power) {
  CHECK_GT(val, GetZeroVal<T>());
  T max_floor = static_cast<T>(std::pow(2, max_power));
  val = std::min(val, max_floor);
  T ret = GetOneVal<T>();
  while (true) {
    ret *= 2;
    if (ret >= val) { return ret == val ? ret : ret / 2; }
  }
}

template<typename T>
T PowOf2Ceil(T val, int32_t max_power) {
  CHECK_GT(val, GetZeroVal<T>());
  T max_ceil = static_cast<T>(std::pow(2, max_power));
  val = std::min(val, max_ceil);
  T ret = GetOneVal<T>();
  while (true) {
    ret *= 2;
    if (ret >= val) { return ret; }
  }
}

template<typename T, typename Compare>
__device__ void BitonicSwap(T* data, const int32_t i, const int32_t j, const bool dir,
                            const Compare& comp) {
  if (comp(data[i], data[j]) == dir) {
    T tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }
}

// https://en.wikipedia.org/wiki/Bitonic_sorter
template<typename T, typename Compare>
__device__ void BitonicSort(T* data, const int32_t elem_cnt, const Compare& comp) {
  // The element count of instance should be pow-of-2
  assert(elem_cnt > 0 && !(elem_cnt & (elem_cnt - 1)));

  // Generate a bitonic sequence from input
  for (int32_t size = 2; size <= elem_cnt / 2; size *= 2) {
    // Merge 2 bitonic sequences of length 'size' into a bitonic sequence of length '2 * size'
    for (int32_t stride = size / 2; stride > 0; stride /= 2) {
      for (int32_t swap_id = threadIdx.x; swap_id < elem_cnt / 2; swap_id += blockDim.x) {
        // Change dir at intervals of 'size / 2' swaps
        const bool dir = swap_id & (size / 2);
        // Locate the pair {pos, pos + stride} which is going te be swaped if needed
        const int pos = 2 * swap_id - (swap_id & (stride - 1));

        BitonicSwap(data, pos, pos + stride, dir, comp);

        __syncthreads();
      }
    }
  }

  // Sort the bitonic sequence
  for (int32_t stride = elem_cnt / 2; stride > 0; stride /= 2) {
    for (int32_t swap_id = threadIdx.x; swap_id < elem_cnt / 2; swap_id += blockDim.x) {
      // Locate the pair {pos, pos + stride} which is going te be swaped if needed
      const int pos = 2 * swap_id - (swap_id & (stride - 1));

      BitonicSwap(data, pos, pos + stride, false, comp);

      __syncthreads();
    }
  }
}

template<typename T>
class Entry final {
 public:
  __device__ __forceinline__ Entry(int32_t index, T value) : index_(index), value_(value) {}

  __device__ __forceinline__ int32_t GetIndex() const { return index_; }
  __device__ __forceinline__ T GetValue() const { return value_; }
  __device__ __forceinline__ void SetIndex(int32_t index) { index_ = index; }
  __device__ __forceinline__ void SetValue(T value) { value_ = value; }

  __device__ __forceinline__ bool operator<(const Entry& entry) const {
    return (value_ < entry.GetValue()) || (value_ == entry.GetValue() && index_ > entry.GetIndex());
  }
  __device__ __forceinline__ bool operator>(const Entry& entry) const {
    return (value_ > entry.GetValue()) || (value_ == entry.GetValue() && index_ < entry.GetIndex());
  }

 private:
  int32_t index_;
  T value_;
};

template<typename T>
class MinHeap final {
 public:
  __device__ __forceinline__ MinHeap(Entry<T>* data, const int32_t heap_size,
                                     const int32_t init_index, const T init_value)
      : data_(data), heap_size_(heap_size) {
    for (int32_t i = 0; i < heap_size; ++i) {
      data_[i].SetIndex(init_index);
      data_[i].SetValue(init_value);
    }
  }
  __device__ __forceinline__ Entry<T>& Top() { return data_[0]; }
  __device__ __forceinline__ void Swap(const int32_t i, const int32_t j) {
    auto tmp = data_[j];
    data_[j] = data_[i];
    data_[i] = tmp;
  }
  __device__ __forceinline__ void MinHeapify(int32_t index) {
    while (true) {
      const int32_t left = 2 * index + 1;
      const int32_t right = 2 * index + 2;
      int32_t min = index;
      if (left < heap_size_ && data_[left] < data_[min]) { min = left; }
      if (right < heap_size_ && data_[right] < data_[min]) { min = right; }
      if (min == index) { return; }
      Swap(min, index);
      index = min;
    }
  }

 private:
  Entry<T>* data_;
  int32_t heap_size_;
};

template<typename T>
__global__ void HeapTopKKernel(const T* in_ptr, const int32_t instance_num,
                               const int32_t instance_size, const int32_t k,
                               const int32_t heap_size, const int32_t init_index,
                               const T init_value, int32_t* out_ptr) {
  extern __shared__ char smem[];
  auto* shared_entries = reinterpret_cast<Entry<T>*>(smem);

  // Divide elements to be sorted into disjoint sets (# of sets == # of heaps).
  // Each thread in the thread block manipulates one heap to select top heap_size entries from
  // corresponding set
  const T* input = in_ptr + blockIdx.x * instance_size;
  auto heap =
      MinHeap<T>(shared_entries + threadIdx.x * heap_size, heap_size, init_index, init_value);
  for (int32_t i = threadIdx.x; i < instance_size; i += blockDim.x) {
    auto entry = Entry<T>(i, input[i]);
    if (entry > heap.Top()) {
      heap.Top() = entry;
      heap.MinHeapify(0);
    }
  }

  __syncthreads();

  // Merge all heaps into a unified, sorted array
  BitonicSort(shared_entries, blockDim.x * heap_size,
              [](const Entry<T>& x, const Entry<T>& y) { return x > y; });

  // Write top_k elements in sorted array to output
  for (int32_t i = threadIdx.x; i < k; i += blockDim.x) {
    (out_ptr + blockIdx.x * k)[i] = shared_entries[i].GetIndex();
  }
}

}  // namespace

template<typename T>
class GpuHeapSelectionTopKKernel final : public user_op::OpKernel {
 public:
  GpuHeapSelectionTopKKernel() = default;
  ~GpuHeapSelectionTopKKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int32_t instance_size = in->shape().At(in->shape().NumAxes() - 1);
    const int32_t instance_num = in->shape().elem_cnt() / instance_size;
    const int32_t k = std::min(ctx->Attr<int32_t>("k"), instance_size);

    // Use as many heaps as possible (# of heaps == # of threads used in thread block).
    // Limitation 1: size of shared memory
    // We also need heap_size * num_heap to be pow-of-2 which is necessary for bitonic sort
    const int32_t heap_size = PowOf2Ceil(k, 16);
    int32_t num_heap =
        PowOf2Floor(kCudaMaxSharedMemoryByteSize / (heap_size * sizeof(Entry<T>)), 16);
    // Limitation 2: # of threads in thread block
    num_heap = std::min(num_heap, kCudaThreadsNumPerBlock);

    HeapTopKKernel<T><<<instance_num, num_heap, num_heap * heap_size * sizeof(Entry<T>),
                        ctx->device_ctx()->cuda_stream()>>>(
        in->dptr<T>(), instance_num, instance_size, k, heap_size, GetMaxVal<int32_t>(),
        GetMinVal<T>(), out->mut_dptr<int32_t>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_HEAP_SELECTION_TOP_K_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("top_k").SetCreateFn<GpuHeapSelectionTopKKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu") & (user_op::HobAttr<int32_t>("k") <= 128)                \
      & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_GPU_HEAP_SELECTION_TOP_K_KERNEL(float)
REGISTER_GPU_HEAP_SELECTION_TOP_K_KERNEL(double)
REGISTER_GPU_HEAP_SELECTION_TOP_K_KERNEL(int32_t)
REGISTER_GPU_HEAP_SELECTION_TOP_K_KERNEL(int64_t)

}  // namespace oneflow
