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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/include/primitive/batch_matmul.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/cuda/atomic.cuh"
#include <mma.h>

namespace oneflow {

namespace {

__global__ void GenerateGatherIndicesGpu(const int32_t elem_cnt, const int32_t stride,
                                         const int32_t in_cols, const int32_t offset,
                                         int32_t* gather_indices) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row = i / stride;
    const int32_t col = i - row * stride;
    if (col < row + offset) {
      int32_t in_index = row * in_cols + col;
      int32_t idx = row * (offset + row - 1 + offset) / 2 + col;
      gather_indices[idx] = in_index;
    }
  }
}

template<typename T>
__global__ void GatherConcatGpu(int32_t elem_cnt, int32_t out_cols, int32_t valid_out_cols,
                                int32_t in_cols, int32_t output_concat_end_dim,
                                const int32_t* gather_indices, const T* in,
                                const T* output_concat_ptr, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row = i / out_cols;
    const int32_t col = i - row * out_cols;
    T out_val;
    if (col < output_concat_end_dim) {
      const int32_t output_concat_idx = row * output_concat_end_dim + col;
      out_val = output_concat_ptr[output_concat_idx];
    } else if (col < valid_out_cols) {
      const int32_t gather_col_idx = gather_indices[col - output_concat_end_dim];
      const int32_t in_offset = row * in_cols + gather_col_idx;
      out_val = in[in_offset];
    } else {
      out_val = 0;
    }
    out_ptr[i] = out_val;
  }
}

template<typename T>
__global__ void ScatterSplitAddTransposeGpu(int32_t elem_cnt, int32_t stride_dim, int32_t out_dim,
                                            int32_t in_grad_stride, int32_t in_grad_matrix_dim,
                                            int32_t in_grad_matrix_valid_dim,
                                            int32_t output_concat_end_dim, const int32_t offset,
                                            const T* dy, T* output_concat_grad, T* in_grad) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row = i / stride_dim;
    const int32_t col = i - row * stride_dim;
    if (col < output_concat_end_dim) {
      output_concat_grad[row * output_concat_end_dim + col] = dy[row * out_dim + col];
    } else {
      int32_t in_col_id = col - output_concat_end_dim;
      const int32_t matrix_row = in_col_id / in_grad_matrix_dim;
      const int32_t matrix_col = in_col_id - matrix_row * in_grad_matrix_dim;
      T grad_val = 0;
      const T* row_dy = dy + row * out_dim + output_concat_end_dim;
      if (matrix_row < in_grad_matrix_valid_dim && matrix_col < in_grad_matrix_valid_dim) {
        if (matrix_col < matrix_row) {
          int32_t dy_col_idx = matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = row_dy[dy_col_idx];
        } else if (matrix_row < matrix_col) {
          // transpose add
          int32_t trans_row_id = matrix_col;
          int32_t trans_col_id = matrix_row;
          int32_t dy_col_idx =
              trans_row_id * (offset + trans_row_id - 1 + offset) / 2 + trans_col_id;
          grad_val = row_dy[dy_col_idx];
        } else if ((matrix_row == matrix_col) && (offset == 1)) {
          int32_t dy_col_idx = matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = row_dy[dy_col_idx] * static_cast<T>(2);
        }
      }
      int32_t in_grad_offset = row * in_grad_stride + in_col_id;
      in_grad[in_grad_offset] = grad_val;
    }
  }
}

template<typename T>
void ConcatFeatures(user_op::KernelComputeContext* ctx, int64_t dst_rows, int64_t dst_cols,
                    void* dst_ptr) {
  const int64_t feature_input_size = ctx->input_size("features");
  auto primitive = ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, 2);
  DimVector dst_shape = {dst_rows, dst_cols};
  int64_t out_col_offset = 0;
  for (int64_t i = 0; i < feature_input_size; ++i) {
    const user_op::Tensor* feature = ctx->Tensor4ArgNameAndIndex("features", i);
    const int64_t feature_rows = feature->shape_view().At(0);
    const int64_t feature_cols = feature->shape_view().Count(1);
    DimVector dst_pos_vec = {0, out_col_offset};
    DimVector src_shape = {feature_rows, feature_cols};
    DimVector src_pos_vec = {0, 0};
    DimVector extent_vec = {feature_rows, feature_cols};
    primitive->Launch(ctx->stream(), feature->data_type(), 2, dst_ptr, dst_shape.data(),
                      dst_pos_vec.data(), feature->dptr<T>(), src_shape.data(), src_pos_vec.data(),
                      extent_vec.data());
    out_col_offset += feature_cols;
  }
  int64_t pad_dim = dst_cols - out_col_offset;
  if (pad_dim > 0) {
    char* out_ptr = reinterpret_cast<char*>(dst_ptr) + out_col_offset * sizeof(T);
    OF_CUDA_CHECK(cudaMemset2DAsync(out_ptr, dst_cols * sizeof(T), 0, pad_dim * sizeof(T), dst_rows,
                                    ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }
}

template<typename T>
void GatherConcatKernel(ep::Stream* stream, int32_t elem_cnt, int32_t out_dim,
                        int32_t valid_out_dim, int32_t features_concated_dim,
                        int32_t concated_padded_dim, int32_t output_concat_end_dim,
                        bool self_interaction, const T* matmul_out, const T* output_concat_ptr,
                        int32_t* gather_indices_ptr, T* out_ptr) {
  cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
  const int32_t gen_indices_elem_cnt = features_concated_dim * features_concated_dim;
  int32_t offset = self_interaction ? 1 : 0;
  GenerateGatherIndicesGpu<<<BlocksNum4ThreadsNum(gen_indices_elem_cnt), kCudaThreadsNumPerBlock, 0,
                             cuda_stream>>>(gen_indices_elem_cnt, features_concated_dim,
                                            concated_padded_dim, offset, gather_indices_ptr);

  int32_t matmul_stride = concated_padded_dim * concated_padded_dim;
  GatherConcatGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
      elem_cnt, out_dim, valid_out_dim, matmul_stride, output_concat_end_dim, gather_indices_ptr,
      matmul_out, output_concat_ptr, out_ptr);
}

template<typename T>
void ScatterSplitAddTranspose(ep::Stream* stream, int32_t batch_size, int32_t out_dim,
                              int32_t concated_padded_dim, int32_t features_concated_dim,
                              int32_t output_concat_end_dim, const bool self_interaction,
                              const T* dy, T* output_concat_grad, T* matmul_out_grad_ptr) {
  int32_t stride_dim = output_concat_end_dim + concated_padded_dim * concated_padded_dim;
  int32_t matmul_stride = concated_padded_dim * concated_padded_dim;
  const int32_t elem_cnt = batch_size * stride_dim;
  int32_t offset = self_interaction ? 1 : 0;
  ScatterSplitAddTransposeGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                stream->As<ep::CudaStream>()->cuda_stream()>>>(
      elem_cnt, stride_dim, out_dim, matmul_stride, concated_padded_dim, features_concated_dim,
      output_concat_end_dim, offset, dy, output_concat_grad, matmul_out_grad_ptr);
}

template<typename T>
void ConcatFeaturesGrad(user_op::KernelComputeContext* ctx, const int64_t batch_size,
                        const int64_t concated_padded_dim, const int64_t vector_size,
                        const T* concated_features_grad) {
  auto primitive = ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, 2);
  DimVector src_shape = {batch_size, concated_padded_dim * vector_size};
  int64_t in_col_offset = 0;
  for (int64_t i = 0; i < ctx->output_size("features_grad"); ++i) {
    user_op::Tensor* feature_grad = ctx->Tensor4ArgNameAndIndex("features_grad", i);
    const int64_t feature_grad_rows = feature_grad->shape_view().At(0);
    const int64_t feature_grad_cols = feature_grad->shape_view().Count(1);
    DimVector dst_shape = {feature_grad_rows, feature_grad_cols};
    DimVector dst_pos_vec = {0, 0};
    DimVector src_pos_vec = {0, in_col_offset};
    DimVector extent_vec = {feature_grad_rows, feature_grad_cols};
    in_col_offset += feature_grad_cols;
    primitive->Launch(ctx->stream(), feature_grad->data_type(), 2, feature_grad->mut_dptr(),
                      dst_shape.data(), dst_pos_vec.data(), concated_features_grad,
                      src_shape.data(), src_pos_vec.data(), extent_vec.data());
  }
}

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

template<typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

int64_t GetPaddedDim(int64_t dim) {
  const int64_t align_dim = 16;
  const int64_t padded_dim = (dim + align_dim - 1) / align_dim * align_dim;
  return padded_dim;
}

template<typename T, int32_t max_in>
struct DotFwdParam {
  const T* in[max_in];
  int32_t in_feature_dim[max_in];
  int32_t dim_start_offset[max_in];
  const T* sparse_feature;
  const uint32_t* sparse_indices;
  int32_t sparse_dim;
  int32_t sparse_dim_start;
  int32_t features_dim;
  const T* output_concat;
  int32_t output_concat_size;
  T* out;
  int32_t num_in;
};

#if __CUDA_ARCH__ >= 700
template<typename T, typename AccType, int m, int n, int k, class ALayout, class BLayout>
class Wmma {
 public:
  __device__ void LoadA(const T* ptr, int ldm) { nvcuda::wmma::load_matrix_sync(a_, ptr, ldm); }
  __device__ void LoadB(const T* ptr, int ldm) { nvcuda::wmma::load_matrix_sync(b_, ptr, ldm); }
  __device__ void Store(AccType* ptr, int ldm) {
    nvcuda::wmma::store_matrix_sync(ptr, acc_, ldm, nvcuda::wmma::mem_row_major);
  }
  __device__ void Mma() { nvcuda::wmma::mma_sync(acc_, a_, b_, acc_); }
  __device__ void InitAcc() { nvcuda::wmma::fill_fragment(acc_, 0.0f); }
  __device__ __forceinline__ T Convert(T src) { return src; }

 private:
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, m, n, k, T, ALayout> a_;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, m, n, k, T, BLayout> b_;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, m, n, k, AccType> acc_;
};

template<typename AccType, int m, int n, int k, class ALayout, class BLayout>
class Wmma<float, AccType, m, n, k, ALayout, BLayout> {
 public:
#if __CUDA_ARCH__ >= 800
  __device__ void LoadA(const float* ptr, int ldm) { nvcuda::wmma::load_matrix_sync(a_, ptr, ldm); }
  __device__ void LoadB(const float* ptr, int ldm) { nvcuda::wmma::load_matrix_sync(b_, ptr, ldm); }
  __device__ void Mma() { nvcuda::wmma::mma_sync(acc_, a_, b_, acc_); }
  __device__ __forceinline__ float Convert(float src) { return nvcuda::wmma::__float_to_tf32(src); }
  __device__ void Store(AccType* ptr, int ldm) {
    nvcuda::wmma::store_matrix_sync(ptr, acc_, ldm, nvcuda::wmma::mem_row_major);
  }
  __device__ void InitAcc() { nvcuda::wmma::fill_fragment(acc_, 0.0f); }
#else
  __device__ void LoadA(const float* ptr, int ldm) { __trap(); }
  __device__ void LoadB(const float* ptr, int ldm) { __trap(); }
  __device__ void Mma() { __trap(); }
  __device__ __forceinline__ float Convert(float src) { return src; }
  __device__ void Store(AccType* ptr, int ldm) { __trap(); }
  __device__ void InitAcc() { __trap(); }
#endif

 private:
#if __CUDA_ARCH__ >= 800
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, m, n, k, nvcuda::wmma::precision::tf32, ALayout>
      a_;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, m, n, k, nvcuda::wmma::precision::tf32, BLayout>
      b_;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, m, n, k, AccType> acc_;
#endif
};
#endif  //__CUDA_ARCH__ >= 700

constexpr int kUnrollDim = 2;
template<typename T, typename ComputeType, int32_t max_in, int32_t pack_size, int mn_tile_dim,
         int k_tile_dim>
__global__ void DotFeatureInteractionWmmaImpl(
    int m_num_tiles, int k_num_tiles, int64_t batch_size, int padded_num_rows, int vector_num_pack,
    int padded_vector_num_pack, int out_num_cols, int out_num_cols_num_pack, int in_shared_mem_cols,
    int in_shared_mem_cols_num_pack, int acc_shared_mem_cols, int acc_shared_mem_cols_num_pack,
    int offset, int output_padding, DotFwdParam<T, max_in> param) {
#if __CUDA_ARCH__ >= 700
  Wmma<T, ComputeType, mn_tile_dim, mn_tile_dim, k_tile_dim, nvcuda::wmma::row_major,
       nvcuda::wmma::col_major>
      wmma;
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  T* buf = reinterpret_cast<T*>(shared_buf);
  Pack<T, pack_size>* buf_pack = reinterpret_cast<Pack<T, pack_size>*>(shared_buf);
  ComputeType* acc_buf =
      reinterpret_cast<ComputeType*>(shared_buf + padded_num_rows * in_shared_mem_cols * sizeof(T));
  int batch_idx = blockIdx.x;
  T* batch_out = param.out + batch_idx * out_num_cols;
  Pack<T, pack_size>* batch_out_pack =
      reinterpret_cast<Pack<T, pack_size>*>(param.out) + batch_idx * out_num_cols_num_pack;
  const int output_concat_size = param.output_concat_size;
  const T* batch_output_concat =
      (param.output_concat) ? (param.output_concat + batch_idx * output_concat_size) : nullptr;
  const uint32_t* batch_sparse_indices =
      (param.sparse_indices) ? (param.sparse_indices + batch_idx * param.sparse_dim) : nullptr;
  const Pack<T, pack_size>* sparse_feature_pack =
      (param.sparse_feature) ? reinterpret_cast<const Pack<T, pack_size>*>(param.sparse_feature)
                             : nullptr;
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
// load dense feature to shared_mem
#pragma unroll
    for (int i = 0; i < max_in; ++i) {
      if (i >= param.num_in) { break; }
      const Pack<T, pack_size>* batch_in = reinterpret_cast<const Pack<T, pack_size>*>(param.in[i])
                                           + batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * kUnrollDim; j < param.in_feature_dim[i];
           j += blockDim.y * kUnrollDim) {
#pragma unroll
        for (int k = 0; k < kUnrollDim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) { break; }
          int buf_row = param.dim_start_offset[i] + in_row;
          Pack<T, pack_size> pack_in_val = batch_in[in_row * vector_num_pack + col];
#pragma unroll
          for (int t = 0; t < pack_size; ++t) {
            pack_in_val.elem[t] = wmma.Convert(pack_in_val.elem[t]);
          }
          buf_pack[buf_row * in_shared_mem_cols_num_pack + col] = pack_in_val;
        }
      }
    }
    // load sparse feature to shared_mem
    for (int j = threadIdx.y * kUnrollDim; j < param.sparse_dim; j += blockDim.y * kUnrollDim) {
#pragma unroll
      for (int k = 0; k < kUnrollDim; ++k) {
        int in_row = j + k;
        if (in_row >= param.sparse_dim) { break; }
        int buf_row = param.sparse_dim_start + in_row;
        int sparse_in_row = batch_sparse_indices[in_row];
        Pack<T, pack_size> pack_in_val = sparse_feature_pack[sparse_in_row * vector_num_pack + col];
#pragma unroll
        for (int t = 0; t < pack_size; ++t) {
          pack_in_val.elem[t] = wmma.Convert(pack_in_val.elem[t]);
        }
        buf_pack[buf_row * in_shared_mem_cols_num_pack + col] = pack_in_val;
      }
    }
  }
  Pack<T, pack_size> zero;
#pragma unroll
  for (int k = 0; k < pack_size; ++k) { zero.elem[k] = wmma.Convert(0); }
  for (int row = threadIdx.y; row < param.features_dim; row += blockDim.y) {
    for (int col = vector_num_pack + threadIdx.x; col < padded_vector_num_pack; col += blockDim.x) {
      buf_pack[row * in_shared_mem_cols_num_pack + col] = zero;
    }
  }
  __syncthreads();
  for (int blocks_id = warp_id; blocks_id < m_num_tiles * m_num_tiles; blocks_id += blockDim.y) {
    int blocks_row = blocks_id / m_num_tiles;
    int blocks_col = blocks_id - blocks_row * m_num_tiles;
    if (blocks_row >= blocks_col) {
      wmma.InitAcc();
      for (int step = 0; step < k_num_tiles; ++step) {
        T* tile_a_ptr = buf + blocks_row * mn_tile_dim * in_shared_mem_cols + step * k_tile_dim;
        T* tile_b_ptr = buf + blocks_col * mn_tile_dim * in_shared_mem_cols + step * k_tile_dim;
        wmma.LoadA(tile_a_ptr, in_shared_mem_cols);
        wmma.LoadB(tile_b_ptr, in_shared_mem_cols);
        wmma.Mma();
      }
      ComputeType* tile_ptr =
          acc_buf + blocks_row * mn_tile_dim * acc_shared_mem_cols + blocks_col * mn_tile_dim;
      wmma.Store(tile_ptr, acc_shared_mem_cols);
    }
  }
  __syncthreads();
  T* emb_out = batch_out + output_concat_size;
  for (int base_row = threadIdx.y * kUnrollDim; base_row < param.features_dim;
       base_row += kUnrollDim * blockDim.y) {
#pragma unroll
    for (int k = 0; k < kUnrollDim; ++k) {
      int row = base_row + k;
      if (row >= param.features_dim) { break; }
      for (int col = threadIdx.x; col < param.features_dim; col += blockDim.x) {
        if (col < row + offset) {
          int64_t idx = row * (offset + row - 1 + offset) / 2 + col;
          emb_out[idx] = static_cast<T>(acc_buf[row * acc_shared_mem_cols + col]);
        }
      }
    }
  }
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = thread_id; i < output_concat_size; i += blockDim.x * blockDim.y) {
    batch_out[i] = batch_output_concat[i];
  }
  for (int i = thread_id; i < output_padding; i += blockDim.x * blockDim.y) {
    batch_out[out_num_cols - 1 - i] = 0;
  }
#else
  __trap();
#endif  // __CUDA_ARCH__ >= 700
}

template<typename T>
struct KTileDim {
  static const int val = 16;
};

template<>
struct KTileDim<float> {
  static const int val = 8;
};

template<typename T, int max_in, int32_t pack_size>
struct DotFeatureInteractionKernel {
  static bool Launch(ep::Stream* stream, int64_t batch_size, int concated_padded_dim,
                     int vector_size, int out_num_cols, bool self_interaction, int output_padding,
                     const DotFwdParam<T, max_in>& param) {
    const int block_size = 128;
    const int block_dim_x = 32;
    const int block_dim_y = block_size / block_dim_x;
    const int num_blocks = batch_size;
    const int mn_tile_dim = 16;
    const int k_tile_dim = KTileDim<T>::val;
    const int64_t padded_vector_size = GetPaddedDim(vector_size);
    const int m_num_tiles = concated_padded_dim / mn_tile_dim;
    const int k_num_tiles = padded_vector_size / k_tile_dim;
    const int skew_in = 8;
    const int skew_acc = 8;
    const int in_shared_mem_num_cols = padded_vector_size + skew_in;
    const int acc_shared_mem_num_cols = concated_padded_dim + skew_acc;
    const size_t in_shared_mem_bytes = concated_padded_dim * in_shared_mem_num_cols * sizeof(T);
    using ComputeType = typename DefaultComputeType<T>::type;
    const size_t acc_shared_mem_bytes =
        concated_padded_dim * acc_shared_mem_num_cols * sizeof(ComputeType);
    const size_t total_shared_mem_bytes = in_shared_mem_bytes + acc_shared_mem_bytes;
    const int32_t offset = self_interaction ? 1 : 0;
    const int out_num_cols_num_pack = out_num_cols / pack_size;
    const int vector_num_pack = vector_size / pack_size;
    const int padded_vector_num_pack = padded_vector_size / pack_size;
    const int in_shared_mem_cols_num_pack = in_shared_mem_num_cols / pack_size;
    const int acc_shared_mem_cols_num_pack = acc_shared_mem_num_cols / pack_size;
    int max_active_blocks;
    OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        DotFeatureInteractionWmmaImpl<T, ComputeType, max_in, pack_size, mn_tile_dim, k_tile_dim>,
        block_size, total_shared_mem_bytes));
    if (max_active_blocks <= 0) { return false; }
    cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
    DotFeatureInteractionWmmaImpl<T, ComputeType, max_in, pack_size, mn_tile_dim, k_tile_dim>
        <<<num_blocks, dim3(block_dim_x, block_dim_y), total_shared_mem_bytes, cuda_stream>>>(
            m_num_tiles, k_num_tiles, batch_size, concated_padded_dim, vector_num_pack,
            padded_vector_num_pack, out_num_cols, out_num_cols_num_pack, in_shared_mem_num_cols,
            in_shared_mem_cols_num_pack, acc_shared_mem_num_cols, acc_shared_mem_cols_num_pack,
            offset, output_padding, param);
    return true;
  }
};

template<typename T, int32_t max_in>
struct DotBwdParam {
  const T* out_grad;
  const T* in[max_in];
  T* in_grad[max_in];
  T* output_concat_grad;
  const T* sparse_feature;
  const uint32_t* sparse_indices;
  int32_t sparse_dim;
  int32_t sparse_dim_start;
  T* sparse_feature_grad;
  int32_t output_concat_size;
  int32_t in_feature_dim[max_in];
  int32_t dim_start_offset[max_in];
  int32_t features_dim;
  int32_t num_in;
};

template<typename T, typename ComputeType, int32_t pack_size>
__device__ __inline__ void AtomicAdd(Pack<T, pack_size>* address,
                                     Pack<ComputeType, pack_size> val) {
#pragma unroll
  for (int i = 0; i < pack_size; ++i) {
    cuda::atomic::Add(reinterpret_cast<T*>(address) + i, static_cast<T>(val.elem[i]));
  }
}

template<>
__device__ __inline__ void AtomicAdd<half, float, 2>(Pack<half, 2>* address, Pack<float, 2> val) {
  half2 h2_val;
  h2_val.x = static_cast<half>(val.elem[0]);
  h2_val.y = static_cast<half>(val.elem[1]);
  cuda::atomic::Add(reinterpret_cast<half2*>(address), h2_val);
}

template<typename T, typename ComputeType, int32_t max_in, int32_t pack_size,
         int32_t sparse_grad_pack_size, int mn_tile_dim, int k_tile_dim>
__global__ void DotFeatureInteractionBackwardWmmaImpl(
    int m_num_tiles, int n_num_tiles, int k_num_tiles, int64_t batch_size, int padded_num_rows,
    int vector_num_pack, int vector_num_sparse_grad_pack, int padded_vector_num_pack,
    int out_num_cols, int in_shared_mem_cols, int in_shared_mem_cols_num_pack,
    int in_shared_mem_cols_num_sparse_grad_pack, int matrix_out_grad_shared_mem_cols, int offset,
    DotBwdParam<T, max_in> param) {
#if __CUDA_ARCH__ >= 700
  Wmma<T, ComputeType, mn_tile_dim, mn_tile_dim, k_tile_dim, nvcuda::wmma::row_major,
       nvcuda::wmma::row_major>
      wmma;
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  T* in_buf = reinterpret_cast<T*>(shared_buf);
  Pack<T, pack_size>* in_buf_pack = reinterpret_cast<Pack<T, pack_size>*>(shared_buf);
  T* matrix_out_grad_buf = in_buf + padded_num_rows * in_shared_mem_cols;
  ComputeType* in_grad_buf = reinterpret_cast<ComputeType*>(
      matrix_out_grad_buf + padded_num_rows * matrix_out_grad_shared_mem_cols);
  Pack<ComputeType, pack_size>* in_grad_buf_pack =
      reinterpret_cast<Pack<ComputeType, pack_size>*>(in_grad_buf);

  int batch_idx = blockIdx.x;
  const T* batch_out_grad = param.out_grad + batch_idx * out_num_cols;
  const int output_concat_size = param.output_concat_size;
  T* batch_output_concat_grad = (param.output_concat_grad)
                                    ? (param.output_concat_grad + batch_idx * output_concat_size)
                                    : nullptr;
  const uint32_t* batch_sparse_indices =
      (param.sparse_indices) ? (param.sparse_indices + batch_idx * param.sparse_dim) : nullptr;
  const Pack<T, pack_size>* sparse_feature_pack =
      (param.sparse_feature) ? reinterpret_cast<const Pack<T, pack_size>*>(param.sparse_feature)
                             : nullptr;

  int features_dim = param.features_dim;
  // 1.split out_grad to concat_out_grad and matrix_out_grad buf
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  for (int i = thread_id; i < output_concat_size; i += blockDim.x * blockDim.y) {
    batch_output_concat_grad[i] = batch_out_grad[i];
  }
  const T* batch_interaction_out_grad = batch_out_grad + output_concat_size;
  for (int matrix_row = threadIdx.y; matrix_row < padded_num_rows; matrix_row += blockDim.y) {
    for (int matrix_col = threadIdx.x; matrix_col < padded_num_rows; matrix_col += blockDim.x) {
      const int64_t i = matrix_row * matrix_out_grad_shared_mem_cols + matrix_col;
      T grad_val = 0;
      if (matrix_row < features_dim && matrix_col < features_dim) {
        if (matrix_col < matrix_row) {
          int32_t out_grad_col = matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = batch_interaction_out_grad[out_grad_col];
        } else if (matrix_row < matrix_col) {
          // transpose add
          int32_t trans_row_id = matrix_col;
          int32_t trans_col_id = matrix_row;
          int32_t out_grad_col =
              trans_row_id * (offset + trans_row_id - 1 + offset) / 2 + trans_col_id;
          grad_val = batch_interaction_out_grad[out_grad_col];
        } else if ((matrix_row == matrix_col) && (offset == 1)) {
          int32_t out_grad_col = matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = batch_interaction_out_grad[out_grad_col] * static_cast<T>(2);
        }
      }
      matrix_out_grad_buf[i] = wmma.Convert(grad_val);
    }
  }

  // 2.load in to in in_buf
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < max_in; ++i) {
      if (i >= param.num_in) { break; }
      const Pack<T, pack_size>* batch_in = reinterpret_cast<const Pack<T, pack_size>*>(param.in[i])
                                           + batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * kUnrollDim; j < param.in_feature_dim[i];
           j += blockDim.y * kUnrollDim) {
#pragma unroll
        for (int k = 0; k < kUnrollDim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) { break; }
          int buf_row = param.dim_start_offset[i] + in_row;
          Pack<T, pack_size> pack_in_val = batch_in[in_row * vector_num_pack + col];
#pragma unroll
          for (int t = 0; t < pack_size; ++t) {
            pack_in_val.elem[t] = wmma.Convert(pack_in_val.elem[t]);
          }
          in_buf_pack[buf_row * in_shared_mem_cols_num_pack + col] = pack_in_val;
        }
      }
    }
    // load sparse feature to shared_mem
    for (int j = threadIdx.y * kUnrollDim; j < param.sparse_dim; j += blockDim.y * kUnrollDim) {
#pragma unroll
      for (int k = 0; k < kUnrollDim; ++k) {
        int in_row = j + k;
        if (in_row >= param.sparse_dim) { break; }
        int buf_row = param.sparse_dim_start + in_row;
        int sparse_in_row = batch_sparse_indices[in_row];
        Pack<T, pack_size> pack_in_val = sparse_feature_pack[sparse_in_row * vector_num_pack + col];
#pragma unroll
        for (int t = 0; t < pack_size; ++t) {
          pack_in_val.elem[t] = wmma.Convert(pack_in_val.elem[t]);
        }
        in_buf_pack[buf_row * in_shared_mem_cols_num_pack + col] = pack_in_val;
      }
    }
  }
  Pack<T, pack_size> zero;
#pragma unroll
  for (int k = 0; k < pack_size; ++k) { zero.elem[k] = wmma.Convert(0); }
#pragma unroll
  for (int row = features_dim + threadIdx.y; row < padded_num_rows; row += blockDim.y) {
    for (int col = threadIdx.x; col < padded_vector_num_pack; col += blockDim.x) {
      in_buf_pack[row * in_shared_mem_cols_num_pack + col] = zero;
    }
  }
  for (int row = threadIdx.y; row < features_dim; row += blockDim.y) {
    for (int col = vector_num_pack + threadIdx.x; col < padded_vector_num_pack; col += blockDim.x) {
      in_buf_pack[row * in_shared_mem_cols_num_pack + col] = zero;
    }
  }
  __syncthreads();

  for (int blocks_id = warp_id; blocks_id < m_num_tiles * n_num_tiles; blocks_id += blockDim.y) {
    int blocks_row = blocks_id / n_num_tiles;
    int blocks_col = blocks_id - blocks_row * n_num_tiles;
    wmma.InitAcc();
    for (int step = 0; step < k_num_tiles; ++step) {
      // blocks_row is a row_id, step is a col_id. blocks_col is b col_id,
      // step is b row_id.
      T* tile_a_ptr = matrix_out_grad_buf
                      + blocks_row * mn_tile_dim * matrix_out_grad_shared_mem_cols
                      + step * k_tile_dim;
      T* tile_b_ptr = in_buf + step * k_tile_dim * in_shared_mem_cols + blocks_col * mn_tile_dim;
      wmma.LoadA(tile_a_ptr, matrix_out_grad_shared_mem_cols);
      wmma.LoadB(tile_b_ptr, in_shared_mem_cols);
      wmma.Mma();
    }
    ComputeType* tile_ptr =
        in_grad_buf + blocks_row * mn_tile_dim * in_shared_mem_cols + blocks_col * mn_tile_dim;
    wmma.Store(tile_ptr, in_shared_mem_cols);
  }
  __syncthreads();

  // 4.split in_grad buf to dx
  // shared_mem to dense dx
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < max_in; ++i) {
      if (i >= param.num_in) { break; }
      Pack<T, pack_size>* batch_in_grad = reinterpret_cast<Pack<T, pack_size>*>(param.in_grad[i])
                                          + batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * kUnrollDim; j < param.in_feature_dim[i];
           j += blockDim.y * kUnrollDim) {
#pragma unroll
        for (int k = 0; k < kUnrollDim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) { break; }
          int buf_row = param.dim_start_offset[i] + in_row;
          Pack<T, pack_size> grad_val;
          Pack<ComputeType, pack_size> buf_grad_val =
              in_grad_buf_pack[buf_row * in_shared_mem_cols_num_pack + col];
#pragma unroll
          for (int t = 0; t < pack_size; ++t) {
            grad_val.elem[t] = static_cast<T>(buf_grad_val.elem[t]);
          }
          batch_in_grad[in_row * vector_num_pack + col] = grad_val;
        }
      }
    }
  }
  // shared_mem to sparse dx, sparse in grad use sparse_grad_pack_size
  Pack<ComputeType, sparse_grad_pack_size>* in_grad_buf_sparse_grad_pack =
      reinterpret_cast<Pack<ComputeType, sparse_grad_pack_size>*>(in_grad_buf);
  Pack<T, sparse_grad_pack_size>* sparse_feature_grad_pack =
      reinterpret_cast<Pack<T, sparse_grad_pack_size>*>(param.sparse_feature_grad);
  for (int col = threadIdx.x; col < vector_num_sparse_grad_pack; col += blockDim.x) {
    for (int j = threadIdx.y * kUnrollDim; j < param.sparse_dim; j += blockDim.y * kUnrollDim) {
#pragma unroll
      for (int k = 0; k < kUnrollDim; ++k) {
        int in_row = j + k;
        if (in_row >= param.sparse_dim) { break; }
        int buf_row = param.sparse_dim_start + in_row;
        int sparse_in_row = batch_sparse_indices[in_row];
        Pack<ComputeType, sparse_grad_pack_size> buf_grad_val =
            in_grad_buf_sparse_grad_pack[buf_row * in_shared_mem_cols_num_sparse_grad_pack + col];
        AtomicAdd<T, ComputeType, sparse_grad_pack_size>(
            sparse_feature_grad_pack + sparse_in_row * vector_num_sparse_grad_pack + col,
            buf_grad_val);
      }
    }
  }

#else
  __trap();
#endif  // __CUDA_ARCH__ >= 700
}

template<typename T, int max_in, int32_t pack_size, int32_t sparse_grad_pack_size>
struct DotFeatureInteractionBackwardKernel {
  static bool Launch(ep::Stream* stream, int64_t batch_size, int concated_padded_dim,
                     int vector_size, int out_num_cols, bool self_interaction,
                     const DotBwdParam<T, max_in>& param) {
    const int block_size = 256;
    const int block_dim_x = 32;
    const int block_dim_y = block_size / block_dim_x;
    const int num_blocks = batch_size;
    const int mn_tile_dim = 16;
    const int k_tile_dim = KTileDim<T>::val;
    const int64_t padded_vector_size = GetPaddedDim(vector_size);
    const int m_num_tiles = concated_padded_dim / mn_tile_dim;
    const int k_num_tiles = concated_padded_dim / k_tile_dim;
    const int n_num_tiles = padded_vector_size / mn_tile_dim;
    const int skew_in = 8;
    const int in_shared_mem_num_cols = padded_vector_size + skew_in;
    const int matrix_out_grad_shared_mem_cols = concated_padded_dim + skew_in;
    const size_t in_shared_mem_bytes = concated_padded_dim * in_shared_mem_num_cols * sizeof(T);
    const size_t matrix_out_grad_shared_mem_bytes =
        concated_padded_dim * matrix_out_grad_shared_mem_cols * sizeof(T);
    using ComputeType = typename DefaultComputeType<T>::type;
    const size_t in_grad_shared_mem_bytes =
        concated_padded_dim * in_shared_mem_num_cols * sizeof(ComputeType);
    const size_t total_shared_mem_bytes =
        in_shared_mem_bytes + matrix_out_grad_shared_mem_bytes + in_grad_shared_mem_bytes;
    const int32_t offset = self_interaction ? 1 : 0;
    const int vector_num_pack = vector_size / pack_size;
    const int padded_vector_num_pack = padded_vector_size / pack_size;
    const int in_shared_mem_cols_num_pack = in_shared_mem_num_cols / pack_size;
    const int vector_num_sparse_grad_pack = vector_size / sparse_grad_pack_size;
    const int in_shared_mem_cols_num_sparse_grad_pack =
        in_shared_mem_num_cols / sparse_grad_pack_size;

    int max_active_blocks;
    OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        DotFeatureInteractionBackwardWmmaImpl<T, ComputeType, max_in, pack_size,
                                              sparse_grad_pack_size, mn_tile_dim, k_tile_dim>,
        block_size, total_shared_mem_bytes));
    if (max_active_blocks <= 0) { return false; }
    cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
    DotFeatureInteractionBackwardWmmaImpl<T, ComputeType, max_in, pack_size, sparse_grad_pack_size,
                                          mn_tile_dim, k_tile_dim>
        <<<num_blocks, dim3(block_dim_x, block_dim_y), total_shared_mem_bytes, cuda_stream>>>(
            m_num_tiles, n_num_tiles, k_num_tiles, batch_size, concated_padded_dim, vector_num_pack,
            vector_num_sparse_grad_pack, padded_vector_num_pack, out_num_cols,
            in_shared_mem_num_cols, in_shared_mem_cols_num_pack,
            in_shared_mem_cols_num_sparse_grad_pack, matrix_out_grad_shared_mem_cols, offset,
            param);

    return true;
  }
};

template<typename T, size_t pack>
__global__ void MemsetGpu(int64_t parallel_num, int64_t vector_size, const uint32_t* num_valid,
                          T* dst) {
  size_t count = 0;
  for (int i = 0; i < parallel_num; ++i) { count += num_valid[i] * vector_size; }
  const size_t pack_count = count / pack;
  Pack<T, pack> pack_value;
  for (int i = 0; i < pack; ++i) { pack_value.elem[i] = static_cast<T>(0); }
  auto* pack_dst = reinterpret_cast<Pack<T, pack>*>(dst);
  CUDA_1D_KERNEL_LOOP_T(size_t, i, pack_count) { pack_dst[i] = pack_value; }
  T* tail_dst = dst + pack_count * pack;
  const size_t tail_count = count - pack_count * pack;
  CUDA_1D_KERNEL_LOOP_T(size_t, i, tail_count) { tail_dst[i] = static_cast<T>(0); }
}

template<typename T, size_t pack>
typename std::enable_if<(pack != 0), void>::type LaunchPackMemsetGpu(cudaStream_t stream,
                                                                     const uint32_t* num_valid,
                                                                     T* ptr, size_t sm_count,
                                                                     int64_t vector_size,
                                                                     int64_t parallel_num) {
  MemsetGpu<T, pack><<<2 * sm_count, 1024, 0, stream>>>(parallel_num, vector_size, num_valid, ptr);
}

template<typename T, size_t pack>
typename std::enable_if<(pack == 0), void>::type LaunchPackMemsetGpu(cudaStream_t stream,
                                                                     const uint32_t* num_valid,
                                                                     T* ptr, size_t sm_count,
                                                                     int64_t vector_size,
                                                                     int64_t parallel_num) {
  LOG(FATAL) << "wrong alignment";
}

template<typename T>
void LaunchMemset(cudaStream_t stream, size_t sm_count, int64_t vector_size, int64_t parallel_num,
                  const uint32_t* num_valid, T* ptr) {
  auto uintptr = reinterpret_cast<std::uintptr_t>(ptr);
  if (uintptr % 16 == 0) {
    LaunchPackMemsetGpu<T, 16 / sizeof(T)>(stream, num_valid, ptr, sm_count, vector_size,
                                           parallel_num);
  } else if (uintptr % 8 == 0) {
    LaunchPackMemsetGpu<T, 8 / sizeof(T)>(stream, num_valid, ptr, sm_count, vector_size,
                                          parallel_num);
  } else if (uintptr % 4 == 0) {
    LaunchPackMemsetGpu<T, 4 / sizeof(T)>(stream, num_valid, ptr, sm_count, vector_size,
                                          parallel_num);
  } else if (uintptr % 2 == 0) {
    LaunchPackMemsetGpu<T, 2 / sizeof(T)>(stream, num_valid, ptr, sm_count, vector_size,
                                          parallel_num);
  } else {
    LaunchPackMemsetGpu<T, 1 / sizeof(T)>(stream, num_valid, ptr, sm_count, vector_size,
                                          parallel_num);
  }
}

template<typename T, int max_in>
bool DispatchFeatureInteractionDotPackSize(user_op::KernelComputeContext* ctx,
                                           const int32_t input_size) {
  CHECK_LE(input_size, max_in) << input_size;
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  const int64_t batch_size = out->shape_view().At(0);
  const int64_t out_num_cols = out->shape_view().At(1);
  const int64_t vector_size = ctx->TensorDesc4ArgNameAndIndex("features", 0)->shape().At(2);
  DotFwdParam<T, max_in> param;
  param.num_in = input_size;
  param.out = out->mut_dptr<T>();
  int64_t features_concated_dim = 0;
  for (int i = 0; i < input_size; ++i) {
    param.in[i] = ctx->Tensor4ArgNameAndIndex("features", i)->dptr<T>();
    param.in_feature_dim[i] = ctx->TensorDesc4ArgNameAndIndex("features", i)->shape().At(1);
    param.dim_start_offset[i] = features_concated_dim;
    features_concated_dim += param.in_feature_dim[i];
  }
  if (ctx->has_input("sparse_feature", 0)) {
    CHECK(ctx->has_input("sparse_indices", 0));
    const user_op::Tensor* sparse_feature = ctx->Tensor4ArgNameAndIndex("sparse_feature", 0);
    const user_op::Tensor* sparse_indices = ctx->Tensor4ArgNameAndIndex("sparse_indices", 0);
    param.sparse_feature = sparse_feature->dptr<T>();
    CHECK_EQ(sparse_indices->data_type(), DataType::kUInt32);
    param.sparse_indices = reinterpret_cast<const uint32_t*>(sparse_indices->dptr());
    param.sparse_dim = ctx->TensorDesc4ArgNameAndIndex("sparse_indices", 0)->shape().At(1);
    param.sparse_dim_start = features_concated_dim;
    features_concated_dim += param.sparse_dim;
  } else {
    param.sparse_feature = nullptr;
    param.sparse_indices = nullptr;
    param.sparse_dim = 0;
    param.sparse_dim_start = 0;
  }
  const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
  param.features_dim = features_concated_dim;
  if (ctx->has_input("output_concat", 0)) {
    const user_op::Tensor* output_concat = ctx->Tensor4ArgNameAndIndex("output_concat", 0);
    param.output_concat = output_concat->dptr<T>();
    param.output_concat_size = output_concat->shape_view().At(1);
  } else {
    param.output_concat = nullptr;
    param.output_concat_size = 0;
  }
  const bool self_interaction = ctx->Attr<bool>("self_interaction");
  const int32_t output_padding = ctx->Attr<int32_t>("output_padding");
  if (vector_size % 4 == 0 && out_num_cols % 4 == 0) {
    return DotFeatureInteractionKernel<T, max_in, 4>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        output_padding, param);
  } else if (vector_size % 2 == 0 && out_num_cols % 2 == 0) {
    return DotFeatureInteractionKernel<T, max_in, 2>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        output_padding, param);
  } else {
    return DotFeatureInteractionKernel<T, max_in, 1>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        output_padding, param);
  }
}

template<typename T, int max_in>
bool DispatchFeatureInteractionDotBackwardPackSize(user_op::KernelComputeContext* ctx,
                                                   const int32_t input_size) {
  CHECK_LE(input_size, max_in) << input_size;
  user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
  const int64_t batch_size = dy->shape_view().At(0);
  const int64_t out_num_cols = dy->shape_view().At(1);
  const int64_t vector_size = ctx->TensorDesc4ArgNameAndIndex("features", 0)->shape().At(2);
  DotBwdParam<T, max_in> param;
  param.num_in = input_size;
  param.out_grad = dy->dptr<T>();
  int64_t features_concated_dim = 0;
  for (int i = 0; i < input_size; ++i) {
    param.in[i] = ctx->Tensor4ArgNameAndIndex("features", i)->dptr<T>();
    param.in_grad[i] = ctx->Tensor4ArgNameAndIndex("features_grad", i)->mut_dptr<T>();
    param.in_feature_dim[i] = ctx->TensorDesc4ArgNameAndIndex("features", i)->shape().At(1);
    param.dim_start_offset[i] = features_concated_dim;
    features_concated_dim += param.in_feature_dim[i];
  }
  if (ctx->has_input("sparse_feature", 0)) {
    CHECK(ctx->has_input("sparse_indices", 0));
    CHECK(ctx->has_input("num_valid_sparse_feature", 0));
    CHECK(ctx->has_output("sparse_feature_grad", 0));
    const user_op::Tensor* sparse_feature = ctx->Tensor4ArgNameAndIndex("sparse_feature", 0);
    const user_op::Tensor* sparse_indices = ctx->Tensor4ArgNameAndIndex("sparse_indices", 0);
    const user_op::Tensor* num_valid_sparse_feature =
        ctx->Tensor4ArgNameAndIndex("num_valid_sparse_feature", 0);
    param.sparse_feature = sparse_feature->dptr<T>();
    CHECK_EQ(sparse_indices->data_type(), DataType::kUInt32);
    param.sparse_indices = reinterpret_cast<const uint32_t*>(sparse_indices->dptr());
    param.sparse_dim = ctx->TensorDesc4ArgNameAndIndex("sparse_indices", 0)->shape().At(1);
    param.sparse_dim_start = features_concated_dim;
    features_concated_dim += param.sparse_dim;
    param.sparse_feature_grad =
        ctx->Tensor4ArgNameAndIndex("sparse_feature_grad", 0)->mut_dptr<T>();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    CHECK_EQ(num_valid_sparse_feature->data_type(), DataType::kUInt32);
    LaunchMemset<T>(ctx->stream()->As<ep::CudaStream>()->cuda_stream(),
                    ctx->stream()->As<ep::CudaStream>()->device_properties().multiProcessorCount,
                    vector_size, parallel_num,
                    reinterpret_cast<const uint32_t*>(num_valid_sparse_feature->dptr())
                        + parallel_id * parallel_num,
                    param.sparse_feature_grad);
  } else {
    param.sparse_feature = nullptr;
    param.sparse_indices = nullptr;
    param.sparse_feature_grad = nullptr;
    param.sparse_dim = 0;
    param.sparse_dim_start = 0;
  }
  const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
  param.features_dim = features_concated_dim;
  if (ctx->has_output("output_concat_grad", 0)) {
    user_op::Tensor* output_concat_grad = ctx->Tensor4ArgNameAndIndex("output_concat_grad", 0);
    param.output_concat_grad = output_concat_grad->mut_dptr<T>();
    param.output_concat_size = output_concat_grad->shape_view().At(1);
  } else {
    param.output_concat_grad = nullptr;
    param.output_concat_size = 0;
  }
  const bool self_interaction = ctx->Attr<bool>("self_interaction");
  if (vector_size % 4 == 0) {
    return DotFeatureInteractionBackwardKernel<T, max_in, 4, 2>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        param);
  } else if (vector_size % 2 == 0) {
    return DotFeatureInteractionBackwardKernel<T, max_in, 2, 2>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        param);
  } else {
    if (ctx->has_input("sparse_feature", 0) && dy->data_type() == DataType::kFloat16) {
      UNIMPLEMENTED()
          << "fused dot interaction backward kernel not support sparse_feature with pack_size 1, "
             "because atomicAdd(half) is too slow";
      return false;
    }
    return DotFeatureInteractionBackwardKernel<T, max_in, 1, 1>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        param);
  }
}

template<typename T, int32_t max_in>
struct Param {
  const T* in[max_in];
  int32_t in_feature_dim[max_in];
  T* out;
  int32_t num_in;
};

template<typename T, int32_t max_in, int32_t pack_size>
__global__ void FeatureInteractionSum(int64_t batch_size, int64_t vector_num_pack,
                                      Param<T, max_in> param) {
  using ComputeType = typename DefaultComputeType<T>::type;
  Pack<T, pack_size>* dst_pack = reinterpret_cast<Pack<T, pack_size>*>(param.out);
  for (int batch_idx = blockIdx.x * blockDim.y + threadIdx.y; batch_idx < batch_size;
       batch_idx += gridDim.x * blockDim.y) {
    Pack<T, pack_size>* batch_out = dst_pack + batch_idx * vector_num_pack;
    for (int col_id = threadIdx.x; col_id < vector_num_pack; col_id += blockDim.x) {
      Pack<ComputeType, pack_size> sum;
      Pack<ComputeType, pack_size> square_sum;
#pragma unroll
      for (int k = 0; k < pack_size; ++k) {
        sum.elem[k] = static_cast<ComputeType>(0);
        square_sum.elem[k] = static_cast<ComputeType>(0);
      }
      for (int i = 0; i < max_in; ++i) {
        if (i >= param.num_in) { break; }
        const Pack<T, pack_size>* batch_in =
            reinterpret_cast<const Pack<T, pack_size>*>(param.in[i])
            + batch_idx * param.in_feature_dim[i] * vector_num_pack;
#pragma unroll
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          Pack<T, pack_size> val = batch_in[j * vector_num_pack + col_id];
#pragma unroll
          for (int k = 0; k < pack_size; ++k) {
            const ComputeType compute_val = static_cast<ComputeType>(val.elem[k]);
            sum.elem[k] += compute_val;
            square_sum.elem[k] += compute_val * compute_val;
          }
        }
      }
      Pack<T, pack_size> out;
#pragma unroll
      for (int k = 0; k < pack_size; ++k) {
        out.elem[k] = static_cast<T>((sum.elem[k] * sum.elem[k] - square_sum.elem[k])
                                     * static_cast<ComputeType>(0.5));
      }
      batch_out[col_id] = out;
    }
  }
}

template<typename T, int32_t max_in>
struct GradParam {
  const T* out_grad;
  const T* in[max_in];
  int32_t in_feature_dim[max_in];
  T* in_grad[max_in];
  int32_t num_in;
};

template<typename T, int32_t max_in>
__global__ void FeatureInteractionSumGrad(int64_t batch_size, int64_t vector_size,
                                          GradParam<T, max_in> param) {
  using ComputeType = typename DefaultComputeType<T>::type;
  for (int batch_idx = blockIdx.x * blockDim.y + threadIdx.y; batch_idx < batch_size;
       batch_idx += gridDim.x * blockDim.y) {
    const T* batch_out_grad = param.out_grad + batch_idx * vector_size;
    for (int col_id = threadIdx.x; col_id < vector_size; col_id += blockDim.x) {
      ComputeType sum = 0;
      for (int i = 0; i < max_in; ++i) {
        if (i >= param.num_in) { break; }
        const T* batch_in = param.in[i] + batch_idx * param.in_feature_dim[i] * vector_size;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          sum += static_cast<ComputeType>(batch_in[j * vector_size + col_id]);
        }
      }
      for (int i = 0; i < max_in; ++i) {
        if (i >= param.num_in) { break; }
        const int64_t in_batch_offset = batch_idx * param.in_feature_dim[i] * vector_size;
        const T* batch_in = param.in[i] + in_batch_offset;
        T* batch_in_grad = param.in_grad[i] + in_batch_offset;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          const int64_t offset = j * vector_size + col_id;
          batch_in_grad[offset] =
              static_cast<T>(static_cast<ComputeType>(batch_out_grad[col_id])
                             * (sum - static_cast<ComputeType>(batch_in[offset])));
        }
      }
    }
  }
}

void GetBlockDims(const int64_t vector_size, int* block_dim_x, int* block_dim_y) {
  const int block_size = 256;
  if (vector_size < block_size) {
    *block_dim_x = std::ceil(static_cast<float>(vector_size) / 8) * 8;
    *block_dim_y = (block_size + *block_dim_x - 1) / *block_dim_x;
  } else {
    *block_dim_x = block_size;
    *block_dim_y = 1;
  }
}

int GetNumBlocks(const int64_t num_instances, const int64_t instance_per_block) {
  int max_blocks = (num_instances + instance_per_block - 1) / instance_per_block;
  return std::min(max_blocks, kCudaMaxBlocksNum);
}

template<typename T, int32_t max_in>
void DispatchFeatureInteractionSumPackSize(ep::Stream* stream, const int64_t batch_size,
                                           const int64_t vector_size,
                                           const Param<T, max_in>& param) {
  int block_dim_x;
  int block_dim_y;
  const int pack_size = (vector_size % 2 == 0) ? 2 : 1;
  const int64_t vector_num_pack = vector_size / pack_size;
  GetBlockDims(vector_num_pack, &block_dim_x, &block_dim_y);
  const int num_blocks = GetNumBlocks(batch_size, block_dim_y);
  dim3 block_dims = dim3(block_dim_x, block_dim_y);
  cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
  if (pack_size == 2) {
    FeatureInteractionSum<T, max_in, 2>
        <<<num_blocks, block_dims, 0, cuda_stream>>>(batch_size, vector_num_pack, param);
  } else {
    FeatureInteractionSum<T, max_in, 1>
        <<<num_blocks, block_dims, 0, cuda_stream>>>(batch_size, vector_num_pack, param);
  }
}

template<typename T, int max_in>
void DispatchFeatureInteractionSumInputSize(user_op::KernelComputeContext* ctx,
                                            const int32_t input_size) {
  CHECK_LE(input_size, max_in) << input_size;
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  const int64_t batch_size = out->shape_view().At(0);
  const int64_t vector_size = out->shape_view().At(1);
  Param<T, max_in> param;
  param.num_in = input_size;
  param.out = out->mut_dptr<T>();
  for (int i = 0; i < input_size; ++i) {
    param.in[i] = ctx->Tensor4ArgNameAndIndex("features", i)->dptr<T>();
    param.in_feature_dim[i] = ctx->TensorDesc4ArgNameAndIndex("features", i)->shape().At(1);
  }
  DispatchFeatureInteractionSumPackSize<T, max_in>(ctx->stream(), batch_size, vector_size, param);
}

template<typename T, int max_in>
void DispatchFeatureInteractionSumGradInputSize(user_op::KernelComputeContext* ctx,
                                                const int32_t input_size) {
  CHECK_LE(input_size, max_in) << input_size;
  const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
  const int64_t batch_size = dy->shape_view().At(0);
  const int64_t vector_size = dy->shape_view().At(1);
  int block_dim_x;
  int block_dim_y;
  GetBlockDims(vector_size, &block_dim_x, &block_dim_y);
  const int num_blocks = GetNumBlocks(batch_size, block_dim_y);
  dim3 block_dims = dim3(block_dim_x, block_dim_y);
  GradParam<T, max_in> param;
  param.num_in = input_size;
  param.out_grad = dy->dptr<T>();
  for (int i = 0; i < input_size; ++i) {
    param.in[i] = ctx->Tensor4ArgNameAndIndex("features", i)->dptr<T>();
    param.in_grad[i] = ctx->Tensor4ArgNameAndIndex("features_grad", i)->mut_dptr<T>();
    param.in_feature_dim[i] = ctx->TensorDesc4ArgNameAndIndex("features_grad", i)->shape().At(1);
  }
  FeatureInteractionSumGrad<T, max_in>
      <<<num_blocks, block_dims, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          batch_size, vector_size, param);
}

}  // namespace

template<typename T>
class FusedDotFeatureInteractionPoolingSumKernel final : public user_op::OpKernel,
                                                         public user_op::CudaGraphSupport {
 public:
  FusedDotFeatureInteractionPoolingSumKernel() = default;
  ~FusedDotFeatureInteractionPoolingSumKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CHECK(!ctx->has_input("sparse_feature", 0)) << "pooling sum, sparse_feature is not supported. ";
    const int input_size = ctx->input_size("features");
    if (input_size == 1) {
      DispatchFeatureInteractionSumInputSize<T, 1>(ctx, input_size);
    } else if (input_size == 2) {
      DispatchFeatureInteractionSumInputSize<T, 2>(ctx, input_size);
    } else if (input_size <= 8) {
      DispatchFeatureInteractionSumInputSize<T, 8>(ctx, input_size);
    } else {
      CHECK_LE(input_size, 128) << "input_size must not greater than 128. ";
      DispatchFeatureInteractionSumInputSize<T, 128>(ctx, input_size);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_DOT_FEATURE_INTERACTION_POOLING_SUM_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("fused_dot_feature_interaction")                                 \
      .SetCreateFn<FusedDotFeatureInteractionPoolingSumKernel<dtype>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobAttr<std::string>("pooling") == "sum"));

REGISTER_FUSED_DOT_FEATURE_INTERACTION_POOLING_SUM_KERNEL(float)
REGISTER_FUSED_DOT_FEATURE_INTERACTION_POOLING_SUM_KERNEL(half)

template<typename T>
bool TryLaunchTensorCoreDotKernel(user_op::KernelComputeContext* ctx) {
  const int input_size = ctx->input_size("features");
  if (input_size == 1) {
    return DispatchFeatureInteractionDotPackSize<T, 1>(ctx, input_size);
  } else if (input_size == 2) {
    return DispatchFeatureInteractionDotPackSize<T, 2>(ctx, input_size);
  } else if (input_size <= 8) {
    return DispatchFeatureInteractionDotPackSize<T, 8>(ctx, input_size);
  } else {
    CHECK_LE(input_size, 128) << "input_size must not greater than 128. ";
    return DispatchFeatureInteractionDotPackSize<T, 128>(ctx, input_size);
  }
}

template<typename T>
bool TryLaunchTensorCoreDotBackwardKernel(user_op::KernelComputeContext* ctx) {
  const int input_size = ctx->input_size("features");
  if (input_size == 1) {
    return DispatchFeatureInteractionDotBackwardPackSize<T, 1>(ctx, input_size);
  } else if (input_size == 2) {
    return DispatchFeatureInteractionDotBackwardPackSize<T, 2>(ctx, input_size);
  } else if (input_size <= 8) {
    return DispatchFeatureInteractionDotBackwardPackSize<T, 8>(ctx, input_size);
  } else {
    CHECK_LE(input_size, 128) << "input_size must not greater than 128. ";
    return DispatchFeatureInteractionDotBackwardPackSize<T, 128>(ctx, input_size);
  }
}
template<typename T>
class FusedDotFeatureInteractionKernel final : public user_op::OpKernel,
                                               public user_op::CudaGraphSupport {
 public:
  FusedDotFeatureInteractionKernel() = default;
  ~FusedDotFeatureInteractionKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType data_type = out->data_type();
    CHECK_LT(out->shape_view().elem_cnt(), GetMaxVal<int32_t>());
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    if ((cuda_stream->device_properties().major >= 7 && data_type == DataType::kFloat16)
        || (cuda_stream->device_properties().major >= 8 && data_type == DataType::kFloat)) {
      bool success = TryLaunchTensorCoreDotKernel<T>(ctx);
      if (success == true) { return; }
    }
    CHECK(!ctx->has_input("sparse_feature", 0)) << "sparse_feature is not supported. ";
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t batch_size = out->shape_view().At(0);
    int64_t features_concated_dim = 0;
    for (int64_t i = 0; i < ctx->input_size("features"); ++i) {
      features_concated_dim += ctx->TensorDesc4ArgNameAndIndex("features", i)->shape().At(1);
    }
    const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
    const int64_t vector_size = ctx->TensorDesc4ArgNameAndIndex("features", 0)->shape().At(2);
    const int64_t out_dim = out->shape_view().At(1);
    const int32_t output_padding = ctx->Attr<int32_t>("output_padding");
    const int64_t valid_out_dim = out_dim - output_padding;
    const bool self_interaction = ctx->Attr<bool>("self_interaction");

    T* matmul_out = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>());
    size_t matmul_out_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * concated_padded_dim * sizeof(T));
    const int64_t interaction_dim = self_interaction
                                        ? features_concated_dim * (features_concated_dim + 1) / 2
                                        : features_concated_dim * (features_concated_dim - 1) / 2;
    int32_t* gather_indices_ptr =
        reinterpret_cast<int32_t*>(tmp_buffer->mut_dptr<char>() + matmul_out_size);
    size_t gather_indices_size = GetCudaAlignedSize(interaction_dim * sizeof(int32_t));
    T* padded_concated_features_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + matmul_out_size + gather_indices_size);
    size_t padded_concated_features_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * vector_size * sizeof(T));
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(),
             matmul_out_size + gather_indices_size + padded_concated_features_size);
    ConcatFeatures<T>(ctx, batch_size, concated_padded_dim * vector_size,
                      padded_concated_features_ptr);
    auto batch_matmul = ep::primitive::NewPrimitive<ep::primitive::BatchMatmulFactory>(
        ctx->device_type(), data_type, ep::primitive::BlasTransposeType::N,
        ep::primitive::BlasTransposeType::T);
    batch_matmul->Launch(ctx->stream(), batch_size, concated_padded_dim, concated_padded_dim,
                         vector_size, 1.0, padded_concated_features_ptr,
                         padded_concated_features_ptr, 0.0, matmul_out);

    int64_t output_concat_end_dim = 0;
    const T* output_concat_ptr = nullptr;
    if (ctx->has_input("output_concat", 0)) {
      user_op::Tensor* output_concat = ctx->Tensor4ArgNameAndIndex("output_concat", 0);
      output_concat_end_dim = output_concat->shape_view().At(1);
      output_concat_ptr = output_concat->dptr<T>();
    }
    CHECK_EQ(valid_out_dim, output_concat_end_dim + interaction_dim);
    GatherConcatKernel<T>(ctx->stream(), out->shape_view().elem_cnt(), out_dim, valid_out_dim,
                          features_concated_dim, concated_padded_dim, output_concat_end_dim,
                          self_interaction, matmul_out, output_concat_ptr, gather_indices_ptr,
                          out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFusedDotFeatureInteractionInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const Shape& first_feature_shape = ctx->InputShape("features", 0);
    const int64_t batch_size = first_feature_shape.At(0);
    const int64_t vector_size = first_feature_shape.At(2);
    int64_t features_concated_dim = 0;
    for (int32_t i = 0; i < ctx->input_size("features"); ++i) {
      features_concated_dim += ctx->InputShape("features", i).At(1);
    }
    const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
    size_t matmul_out_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * concated_padded_dim * sizeof(T));
    const bool self_interaction = ctx->Attr<bool>("self_interaction");
    const int64_t interaction_dim = self_interaction
                                        ? features_concated_dim * (features_concated_dim + 1) / 2
                                        : features_concated_dim * (features_concated_dim - 1) / 2;
    size_t gather_indices_size = GetCudaAlignedSize(interaction_dim * sizeof(int32_t));
    size_t padded_concated_features_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * vector_size * sizeof(T));
    return matmul_out_size + gather_indices_size + padded_concated_features_size;
  };
}

#define REGISTER_FUSED_DOT_FEATURE_INTERACTION_KERNEL(dtype)                            \
  REGISTER_USER_KERNEL("fused_dot_feature_interaction")                                 \
      .SetCreateFn<FusedDotFeatureInteractionKernel<dtype>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobAttr<std::string>("pooling") == "none"))         \
      .SetInferTmpSizeFn(GenFusedDotFeatureInteractionInferTmpSizeFn<dtype>());

REGISTER_FUSED_DOT_FEATURE_INTERACTION_KERNEL(float)
REGISTER_FUSED_DOT_FEATURE_INTERACTION_KERNEL(half)

template<typename T>
class FusedDotFeatureInteractionGradKernel final : public user_op::OpKernel,
                                                   public user_op::CudaGraphSupport {
 public:
  FusedDotFeatureInteractionGradKernel() = default;
  ~FusedDotFeatureInteractionGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const DataType data_type = dy->data_type();
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    if ((cuda_stream->device_properties().major >= 7 && data_type == DataType::kFloat16)
        || (cuda_stream->device_properties().major >= 8 && data_type == DataType::kFloat)) {
      bool success = TryLaunchTensorCoreDotBackwardKernel<T>(ctx);
      if (success == true) { return; }
    }
    CHECK(!ctx->has_input("sparse_feature", 0)) << "sparse_feature is not supported. ";
    const int64_t batch_size = dy->shape_view().At(0);
    int64_t features_concated_dim = 0;
    for (int32_t i = 0; i < ctx->output_size("features_grad"); ++i) {
      features_concated_dim += ctx->TensorDesc4ArgNameAndIndex("features_grad", i)->shape().At(1);
    }
    const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
    const int64_t vector_size = ctx->TensorDesc4ArgNameAndIndex("features_grad", 0)->shape().At(2);
    const int64_t out_dim = dy->shape_view().At(1);
    const bool self_interaction = ctx->Attr<bool>("self_interaction");
    T* matmul_out_grad_ptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>());
    size_t matmul_out_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * concated_padded_dim * sizeof(T));
    T* padded_concated_features_grad_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + matmul_out_grad_size);
    size_t padded_concated_features_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * vector_size * sizeof(T));
    T* padded_concated_features_ptr = reinterpret_cast<T*>(
        tmp_buffer->mut_dptr<char>() + matmul_out_grad_size + padded_concated_features_grad_size);
    size_t padded_concated_features_size = padded_concated_features_grad_size;
    CHECK_LE(
        matmul_out_grad_size + padded_concated_features_grad_size + padded_concated_features_size,
        tmp_buffer->shape_view().elem_cnt());
    ConcatFeatures<T>(ctx, batch_size, concated_padded_dim * vector_size,
                      padded_concated_features_ptr);

    T* output_concat_grad_ptr = nullptr;
    int64_t output_concat_end_dim = 0;
    if (ctx->has_output("output_concat_grad", 0)) {
      user_op::Tensor* output_concat_grad = ctx->Tensor4ArgNameAndIndex("output_concat_grad", 0);
      output_concat_grad_ptr = output_concat_grad->mut_dptr<T>();
      output_concat_end_dim = output_concat_grad->shape_view().At(1);
    }
    ScatterSplitAddTranspose(ctx->stream(), batch_size, out_dim, concated_padded_dim,
                             features_concated_dim, output_concat_end_dim, self_interaction,
                             dy->dptr<T>(), output_concat_grad_ptr, matmul_out_grad_ptr);

    auto batch_matmul = ep::primitive::NewPrimitive<ep::primitive::BatchMatmulFactory>(
        ctx->device_type(), data_type, ep::primitive::BlasTransposeType::N,
        ep::primitive::BlasTransposeType::N);
    batch_matmul->Launch(ctx->stream(), batch_size, concated_padded_dim, vector_size,
                         concated_padded_dim, 1.0, matmul_out_grad_ptr,
                         padded_concated_features_ptr, 0.0, padded_concated_features_grad_ptr);

    ConcatFeaturesGrad(ctx, batch_size, concated_padded_dim, vector_size,
                       padded_concated_features_grad_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFusedDotFeatureInteractionGradInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    int64_t features_concated_dim = 0;
    for (int32_t i = 0; i < ctx->output_size("features_grad"); ++i) {
      features_concated_dim += ctx->InputShape("features_grad", i).At(1);
    }
    const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
    const int64_t batch_size = ctx->InputShape("features_grad", 0).At(0);
    const int64_t vector_size = ctx->InputShape("features_grad", 0).At(2);
    size_t matmul_out_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * concated_padded_dim * sizeof(T));
    size_t padded_concated_features_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * vector_size * sizeof(T));
    size_t padded_concated_features_size = padded_concated_features_grad_size;
    return matmul_out_grad_size + padded_concated_features_grad_size
           + padded_concated_features_size;
  };
}

#define REGISTER_FUSED_DOT_FEATURE_INTERACTION_GRAD_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("fused_dot_feature_interaction_grad")                           \
      .SetCreateFn<FusedDotFeatureInteractionGradKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobAttr<std::string>("pooling") == "none"))        \
      .SetInferTmpSizeFn(GenFusedDotFeatureInteractionGradInferTmpSizeFn<dtype>());

REGISTER_FUSED_DOT_FEATURE_INTERACTION_GRAD_KERNEL(float)
REGISTER_FUSED_DOT_FEATURE_INTERACTION_GRAD_KERNEL(half)

template<typename T>
class FusedDotFeatureInteractionPoolingSumGradKernel final : public user_op::OpKernel,
                                                             public user_op::CudaGraphSupport {
 public:
  FusedDotFeatureInteractionPoolingSumGradKernel() = default;
  ~FusedDotFeatureInteractionPoolingSumGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int input_size = ctx->input_size("features");
    if (input_size == 1) {
      DispatchFeatureInteractionSumGradInputSize<T, 1>(ctx, input_size);
    } else if (input_size == 2) {
      DispatchFeatureInteractionSumGradInputSize<T, 2>(ctx, input_size);
    } else if (input_size <= 8) {
      DispatchFeatureInteractionSumGradInputSize<T, 8>(ctx, input_size);
    } else {
      CHECK_LE(input_size, 128) << "input_size must not greater than 128. ";
      DispatchFeatureInteractionSumGradInputSize<T, 128>(ctx, input_size);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_DOT_FEATURE_INTERACTION_POOLING_SUM_GRAD_KERNEL(dtype)          \
  REGISTER_USER_KERNEL("fused_dot_feature_interaction_grad")                           \
      .SetCreateFn<FusedDotFeatureInteractionPoolingSumGradKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobAttr<std::string>("pooling") == "sum"));

REGISTER_FUSED_DOT_FEATURE_INTERACTION_POOLING_SUM_GRAD_KERNEL(float)
REGISTER_FUSED_DOT_FEATURE_INTERACTION_POOLING_SUM_GRAD_KERNEL(half)

}  // namespace oneflow
