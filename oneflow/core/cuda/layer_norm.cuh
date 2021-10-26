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

#ifndef ONEFLOW_CORE_CUDA_LAYER_NORM_H_
#define ONEFLOW_CORE_CUDA_LAYER_NORM_H_

#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>

namespace oneflow {

namespace cuda {

namespace layer_norm {

constexpr int kWarpSize = 32;

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}
inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
                                int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
  return cudaSuccess;
}

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template<typename SRC, typename DST>
struct DirectLoad {
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = row * row_size + col;
    pack.storage = *reinterpret_cast<const PackType<SRC, N>*>(src + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  const SRC* src;
  int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = row * row_size + col;
#pragma unroll
    for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
    *reinterpret_cast<PackType<DST, N>*>(dst + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
};

template<typename T>
struct WelfordAggregate {
  T mean;
  T m2;
  T count;
  //__device__ WelfordAggregate() : mean(0), m2(0), count(0) {}
  __device__ WelfordAggregate() {}
  __device__ WelfordAggregate(T mean, T m2, T count) : mean(mean), m2(m2), count(count) {}
  inline __device__ void reduce(T val) {
    // Use Welford Online algorithem to compute mean and variance
    // For more details you can refer to:
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    count = count + 1;
    T delta1 = val - mean;
    mean += Div(delta1, count);
    T delta2 = val - mean;
    m2 += delta1 * delta2;
  }
  inline __device__ void combine(WelfordAggregate<T> data) {
    if (data.count == 0) { return; }
    T new_count = count + data.count;
    T nb_over_n = Div(data.count, new_count);
    T delta = data.mean - mean;
    mean += delta * nb_over_n;
    m2 += data.m2 + delta * delta * count * nb_over_n;
    count = new_count;
  }
  inline __device__ T project() const { return Div(m2, count); }
};

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ WelfordAggregate<T> WelfordAggregateWarpAllReduce(WelfordAggregate<T> val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    WelfordAggregate<T> shfl_xor_val(__shfl_xor_sync(0xffffffff, val.mean, mask),
                                     __shfl_xor_sync(0xffffffff, val.m2, mask),
                                     __shfl_xor_sync(0xffffffff, val.count, mask));
    val.combine(shfl_xor_val);
  }
  return val;
}

template<typename T>
__inline__ __device__ WelfordAggregate<T> WelfordAggregateWarpReduce(WelfordAggregate<T> val) {
  for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
    WelfordAggregate<T> shfl_down_val(__shfl_down_sync(0xffffffff, val.mean, mask),
                                      __shfl_down_sync(0xffffffff, val.m2, mask),
                                      __shfl_down_sync(0xffffffff, val.count, mask));
    val.combine(shfl_down_val);
  }
  return val;
}

template<typename T>
__inline__ __device__ WelfordAggregate<T> WelfordAggregateBlockAllReduce(WelfordAggregate<T> val,
                                                                         T* mean_shared,
                                                                         T* m2_shared,
                                                                         T* count_shared) {
  __shared__ WelfordAggregate<T> result_broadcast;
  const int lid = threadIdx.x % 32;
  const int wid = threadIdx.x / 32;
  val = WelfordAggregateWarpReduce(val);
  __syncthreads();
  if (lid == 0) {
    mean_shared[wid] = val.mean;
    m2_shared[wid] = val.m2;
    count_shared[wid] = val.count;
  }
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < blockDim.x / 32) {
      val.mean = mean_shared[lid];
      val.m2 = m2_shared[lid];
      val.count = count_shared[lid];
    } else {
      val.mean = static_cast<T>(0);
      val.m2 = static_cast<T>(0);
      val.count = static_cast<T>(0);
    }
    __syncwarp();
    val = WelfordAggregateWarpReduce(val);
    if (lid == 0) { result_broadcast = val; }
  }
  __syncthreads();
  return result_broadcast;
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                  const double epsilon, ComputeType* mean,
                                  ComputeType* inv_variance) {
  static_assert(cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][cols_per_thread];
  using WelfordType = WelfordAggregate<ComputeType>;
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows;
       row += num_global_thread_group * rows_per_access) {
    WelfordType weldata[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      weldata[row_id] = WelfordType(0, 0, 0);
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          load.template load<pack_size>(row_buf + pack_id * pack_size, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            weldata[row_id].reduce(row_buf[pack_id * pack_size + i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_id * pack_size + i] = 0; }
        }
      }
    }
    WelfordType warp_welford[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      ComputeType* row_buf = buf[row_id];
      warp_welford[row_id] =
          WelfordAggregateWarpAllReduce<ComputeType, thread_group_width>(weldata[row_id]);
      ComputeType row_mean = warp_welford[row_id].mean;
      ComputeType row_variance = max(warp_welford[row_id].project(), static_cast<ComputeType>(0.0));
      ComputeType row_inv_var = rsqrt(row_variance + static_cast<ComputeType>(epsilon));
      if (lane_id == 0) {
        mean[row + row_id] = row_mean;
        inv_variance[row + row_id] = row_inv_var;
      }
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = (row_buf[i] - row_mean) * row_inv_var;
      }
#pragma unroll
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          store.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
        }
      }
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
inline cudaError_t LaunchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols,
                                           const double epsilon, ComputeType* mean,
                                           ComputeType* inv_variance) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int rows_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, rows_per_block);
  const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                    rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access>
inline cudaError_t DispatchLayerNormWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols,
                                                    const double epsilon, ComputeType* mean,
                                                    ComputeType* inv_variance) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                   thread_group_width, rows_per_access, false>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  } else {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                   thread_group_width, rows_per_access, true>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    if (rows % 2 == 0) {                                                                      \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    if (rows % 2 == 0) {                                                                      \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}
template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 4, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    if (rows % 2 == 0) {                                                                      \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(32)
  DEFINE_ONE_ELIF(64)
  DEFINE_ONE_ELIF(128)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
struct DispatchLayerNormWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance) {
    if (cols % 4 == 0) {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 4>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else if (cols % 2 == 0) {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                             const int64_t rows, const int64_t cols,
                                             const double epsilon, ComputeType* mean,
                                             ComputeType* inv_variance) {
  return DispatchLayerNormWarpImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LayerNormBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                       const int64_t cols, const double epsilon, ComputeType* mean,
                                       ComputeType* inv_variance) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  using WelfordType = WelfordAggregate<ComputeType>;
  __shared__ typename std::aligned_storage<sizeof(ComputeType), alignof(ComputeType)>::type
      mean_shared[32];
  ComputeType* mean_shared_ptr = reinterpret_cast<ComputeType*>(mean_shared);
  __shared__
      typename std::aligned_storage<sizeof(ComputeType), alignof(ComputeType)>::type m2_shared[32];
  ComputeType* m2_shared_ptr = reinterpret_cast<ComputeType*>(m2_shared);
  __shared__ typename std::aligned_storage<sizeof(ComputeType), alignof(ComputeType)>::type
      count_shared[32];
  ComputeType* count_shared_ptr = reinterpret_cast<ComputeType*>(count_shared);

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    WelfordType weldata(0, 0, 0);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        weldata.reduce(pack[i]);
      }
    }
    WelfordType block_welford = WelfordAggregateBlockAllReduce<ComputeType>(
        weldata, mean_shared_ptr, m2_shared_ptr, count_shared_ptr);
    ComputeType row_mean = block_welford.mean;
    ComputeType row_variance = max(block_welford.project(), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      mean[row] = row_mean;
      inv_variance[row] = row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = (buf[i * num_packs + pack_id] - row_mean) * row_inv_var;
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
inline cudaError_t LaunchLayerNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                int smem, const int64_t rows, const int64_t cols,
                                                const double epsilon, ComputeType* mean,
                                                ComputeType* inv_variance) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols, epsilon, mean,
                                                 inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline cudaError_t TryDispatchLayerNormBlockSMemImplBlockSize(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType);
  int max_active_blocks_conf_1;

  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>,
        block_size_conf_1, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4>(
        stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3>(
        stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2>(
        stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
  }
  *success = true;
  return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>(
      stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType>
struct TryDispatchLayerNormBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance, bool* success) {
    if (cols % 4 == 0) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 4>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    } else if (cols % 2 == 0) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    } else {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t TryDispatchLayerNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                     const int64_t rows, const int64_t cols,
                                                     const double epsilon, ComputeType* mean,
                                                     ComputeType* inv_variance, bool* success) {
  return TryDispatchLayerNormBlockSMemImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LayerNormBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                           const int64_t cols, const double epsilon,
                                           ComputeType* mean, ComputeType* inv_variance) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  using WelfordType = WelfordAggregate<ComputeType>;
  __shared__ typename std::aligned_storage<sizeof(ComputeType), alignof(ComputeType)>::type
      mean_shared[32];
  ComputeType* mean_shared_ptr = reinterpret_cast<ComputeType*>(mean_shared);
  __shared__
      typename std::aligned_storage<sizeof(ComputeType), alignof(ComputeType)>::type m2_shared[32];
  ComputeType* m2_shared_ptr = reinterpret_cast<ComputeType*>(m2_shared);
  __shared__ typename std::aligned_storage<sizeof(ComputeType), alignof(ComputeType)>::type
      count_shared[32];
  ComputeType* count_shared_ptr = reinterpret_cast<ComputeType*>(count_shared);
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    WelfordType weldata(0, 0, 0);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { weldata.reduce(pack[i]); }
    }
    WelfordType block_welford = WelfordAggregateBlockAllReduce<ComputeType>(
        weldata, mean_shared_ptr, m2_shared_ptr, count_shared_ptr);
    ComputeType row_mean = block_welford.mean;
    ComputeType row_variance = max(block_welford.project(), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      mean[row] = row_mean;
      inv_variance[row] = row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { pack[i] = (pack[i] - row_mean) * row_inv_var; }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline cudaError_t LaunchLayerNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols,
                                                    const double epsilon, ComputeType* mean,
                                                    ComputeType* inv_variance) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType>
struct DispatchLayerNormBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance) {
    if (cols % 4 == 0) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 4>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else if (cols % 2 == 0) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                      const int64_t rows, const int64_t cols,
                                                      const double epsilon, ComputeType* mean,
                                                      ComputeType* inv_variance) {
  return DispatchLayerNormBlockUncachedImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNorm(cudaStream_t stream, LOAD load, STORE store,
                                     const int64_t rows, const int64_t cols, const double epsilon,
                                     ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 1024) {
    return DispatchLayerNormWarpImpl<LOAD, STORE, ComputeType>(stream, load, store, rows, cols,
                                                               epsilon, mean, inv_variance);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err = TryDispatchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance,
          &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
    return cudaSuccess;
  }
}

}  // namespace layer_norm

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_LAYER_NORM_H_
