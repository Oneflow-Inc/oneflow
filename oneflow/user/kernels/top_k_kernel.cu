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
#include "oneflow/user/kernels/radix_sort.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
T PowOf2Floor(T val, int64_t max_power) {
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
T PowOf2Ceil(T val, int64_t max_power) {
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
__device__ void BitonicSwap(T* data, const int64_t i, const int64_t j, const bool dir,
                            const Compare& comp) {
  if (comp(data[i], data[j]) == dir) {
    T tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }
}

// https://en.wikipedia.org/wiki/Bitonic_sorter
template<typename T, typename Compare>
__device__ void BitonicSort(T* data, const int64_t elem_cnt, const Compare& comp) {
  // The element count of instance should be pow-of-2
  assert(elem_cnt > 0 && !(elem_cnt & (elem_cnt - 1)));

  // Generate a bitonic sequence from input
  for (int64_t size = 2; size <= elem_cnt / 2; size *= 2) {
    // Merge 2 bitonic sequences of length 'size' into a bitonic sequence of length '2 * size'
    for (int64_t stride = size / 2; stride > 0; stride /= 2) {
      for (int64_t swap_id = threadIdx.x; swap_id < elem_cnt / 2; swap_id += blockDim.x) {
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
  for (int64_t stride = elem_cnt / 2; stride > 0; stride /= 2) {
    for (int64_t swap_id = threadIdx.x; swap_id < elem_cnt / 2; swap_id += blockDim.x) {
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
  __device__ __forceinline__ Entry(int64_t index, T value) : index_(index), value_(value) {}

  __device__ __forceinline__ int64_t GetIndex() const { return index_; }
  __device__ __forceinline__ T GetValue() const { return value_; }
  __device__ __forceinline__ void SetIndex(int64_t index) { index_ = index; }
  __device__ __forceinline__ void SetValue(T value) { value_ = value; }

  __device__ __forceinline__ bool operator<(const Entry& entry) const {
    return (value_ < entry.GetValue()) || (value_ == entry.GetValue() && index_ > entry.GetIndex());
  }
  __device__ __forceinline__ bool operator>(const Entry& entry) const {
    return (value_ > entry.GetValue()) || (value_ == entry.GetValue() && index_ < entry.GetIndex());
  }

 private:
  int64_t index_;
  T value_;
};

template<typename T>
class MinHeap final {
 public:
  __device__ __forceinline__ MinHeap(Entry<T>* data, const int64_t heap_size,
                                     const int64_t init_index, const T init_value)
      : data_(data), heap_size_(heap_size) {
    for (int64_t i = 0; i < heap_size; ++i) {
      data_[i].SetIndex(init_index);
      data_[i].SetValue(init_value);
    }
  }
  __device__ __forceinline__ Entry<T>& Top() { return data_[0]; }
  __device__ __forceinline__ void Swap(const int64_t i, const int64_t j) {
    auto tmp = data_[j];
    data_[j] = data_[i];
    data_[i] = tmp;
  }
  __device__ __forceinline__ void MinHeapify(int64_t index) {
    while (true) {
      const int64_t left = 2 * index + 1;
      const int64_t right = 2 * index + 2;
      int64_t min = index;
      if (left < heap_size_ && data_[left] < data_[min]) { min = left; }
      if (right < heap_size_ && data_[right] < data_[min]) { min = right; }
      if (min == index) { return; }
      Swap(min, index);
      index = min;
    }
  }

 private:
  Entry<T>* data_;
  int64_t heap_size_;
};

template<typename T>
__global__ void HeapTopKKernel(const T* in_ptr, const int64_t instance_num,
                               const int64_t instance_size, const int64_t k,
                               const int64_t heap_size, const int64_t init_index,
                               const T init_value, int64_t* out_ptr) {
  extern __shared__ char smem[];
  auto* shared_entries = reinterpret_cast<Entry<T>*>(smem);

  // Divide elements to be sorted into disjoint sets (# of sets == # of heaps).
  // Each thread in the thread block manipulates one heap to select top heap_size entries from
  // corresponding set
  const T* input = in_ptr + blockIdx.x * instance_size;
  auto heap =
      MinHeap<T>(shared_entries + threadIdx.x * heap_size, heap_size, init_index, init_value);
  for (int64_t i = threadIdx.x; i < instance_size; i += blockDim.x) {
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
  for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
    (out_ptr + blockIdx.x * k)[i] = shared_entries[i].GetIndex();
  }
}

template<typename T>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(int64_t capacity, void* ptr, const ShapeView& in_shape)
      : capacity_{capacity},
        sorted_in_elem_cnt_{in_shape.elem_cnt()},
        indices_elem_cnt_{sorted_in_elem_cnt_},
        sorted_indices_elem_cnt_{sorted_in_elem_cnt_} {
    const int64_t sorted_in_aligned_bytes = GetCudaAlignedSize(sorted_in_elem_cnt_ * sizeof(T));
    const int64_t indices_aligned_bytes = GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int64_t));
    const int64_t sorted_indices_aligned_bytes = indices_aligned_bytes;
    sorted_in_ptr_ = reinterpret_cast<T*>(ptr);
    indices_ptr_ = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(sorted_in_ptr_)
                                              + sorted_in_aligned_bytes);
    sorted_indices_ptr_ =
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_ptr_ = reinterpret_cast<void*>(reinterpret_cast<char*>(sorted_indices_ptr_)
                                                + sorted_indices_aligned_bytes);
    temp_storage_bytes_ =
        capacity_ - sorted_in_aligned_bytes - indices_aligned_bytes - sorted_indices_aligned_bytes;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  ~TmpBufferManager() = default;

  T* SortedInPtr() const { return sorted_in_ptr_; }
  int64_t* IndicesPtr() const { return indices_ptr_; }
  int64_t* SortedIndicesPtr() const { return sorted_indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int64_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int64_t capacity_;

  T* sorted_in_ptr_;
  int64_t* indices_ptr_;
  int64_t* sorted_indices_ptr_;
  void* temp_storage_ptr_;

  int64_t sorted_in_elem_cnt_;
  int64_t indices_elem_cnt_;
  int64_t sorted_indices_elem_cnt_;
  int64_t temp_storage_bytes_;
};

__global__ void InitializeIndices(int64_t elem_cnt, int64_t* indices_ptr, int64_t instance_size) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { indices_ptr[i] = i % instance_size; };
}

}  // namespace

template<typename T>
class GpuTopKKernel final : public user_op::OpKernel {
 public:
  GpuTopKKernel() = default;
  ~GpuTopKKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    if (in->shape_view().elem_cnt() == 0) { return; }
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t elem_cnt = in->shape_view().elem_cnt();
    const int64_t instance_size = in->shape_view().At(in->shape_view().NumAxes() - 1);
    const int64_t instance_num = elem_cnt / instance_size;
    const int64_t k = std::min(static_cast<int64_t>(ctx->Attr<int32_t>("k")), instance_size);

    if (k > 30 || instance_num == 1) {
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      TmpBufferManager<T> buf_manager(static_cast<int64_t>(tmp_buffer->shape_view().elem_cnt()),
                                      tmp_buffer->mut_dptr<void>(), in->shape_view());

      InitializeIndices<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, buf_manager.IndicesPtr(), instance_size);
      SortPairsDescending(in->dptr<T>(), buf_manager.IndicesPtr(), instance_num, instance_size,
                          buf_manager.TempStoragePtr(), buf_manager.TempStorageBytes(),
                          buf_manager.SortedInPtr(), buf_manager.SortedIndicesPtr(),
                          ctx->stream()->As<ep::CudaStream>()->cuda_stream());
      OF_CUDA_CHECK(cudaMemcpy2DAsync(
          out->mut_dptr<int64_t>(), k * sizeof(int64_t), buf_manager.SortedIndicesPtr(),
          instance_size * sizeof(int64_t), k * sizeof(int64_t), instance_num, cudaMemcpyDefault,
          ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    } else {
      // Use as many heaps as possible (# of heaps == # of threads used in thread block).
      // Limitation 1: size of shared memory
      // We also need heap_size * num_heap to be pow-of-2 which is necessary for bitonic sort
      const int64_t heap_size = PowOf2Ceil(k, 16);
      int32_t num_heap =
          PowOf2Floor(kCudaMaxSharedMemoryByteSize / (heap_size * sizeof(Entry<T>)), 16);
      // Limitation 2: # of threads in thread block
      num_heap = std::min(num_heap, kCudaThreadsNumPerBlock);

      HeapTopKKernel<T><<<instance_num, num_heap, num_heap * heap_size * sizeof(Entry<T>),
                          ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          in->dptr<T>(), instance_num, instance_size, k, heap_size, GetMaxVal<int64_t>(),
          GetMinVal<T>(), out->mut_dptr<int64_t>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_TOP_K_KERNEL(dtype)                                                        \
  REGISTER_USER_KERNEL("top_k")                                                                  \
      .SetCreateFn<GpuTopKKernel<dtype>>()                                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                           \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))          \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const Shape& in_shape = ctx->InputShape("in", 0);                                        \
        const int64_t elem_cnt = in_shape.elem_cnt();                                            \
        const int64_t instance_size = in_shape.dim_vec().back();                                 \
        const int64_t instance_num = elem_cnt / instance_size;                                   \
                                                                                                 \
        /* Sorted In*/                                                                           \
        const int64_t sorted_in_aligned_bytes = GetCudaAlignedSize(elem_cnt * sizeof(dtype));    \
        /* Indices */                                                                            \
        const int64_t indices_aligned_bytes = GetCudaAlignedSize(elem_cnt * sizeof(int64_t));    \
        /* Sorted Indices */                                                                     \
        const int64_t sorted_indices_aligned_bytes = indices_aligned_bytes;                      \
        /* CUB Temp Storage */                                                                   \
        int64_t temp_storage_bytes =                                                             \
            InferTempStorageForSortPairsDescending<dtype, int64_t>(instance_num, instance_size); \
                                                                                                 \
        return sorted_in_aligned_bytes + indices_aligned_bytes + sorted_indices_aligned_bytes    \
               + temp_storage_bytes;                                                             \
      });

REGISTER_CUDA_TOP_K_KERNEL(float)
REGISTER_CUDA_TOP_K_KERNEL(double)
REGISTER_CUDA_TOP_K_KERNEL(uint8_t)
REGISTER_CUDA_TOP_K_KERNEL(int8_t)
REGISTER_CUDA_TOP_K_KERNEL(int32_t)
REGISTER_CUDA_TOP_K_KERNEL(int64_t)
REGISTER_CUDA_TOP_K_KERNEL(half)

}  // namespace oneflow
