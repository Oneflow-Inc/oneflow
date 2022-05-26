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
    const int64_t feature_rows = feature->shape().At(0);
    const int64_t feature_cols = feature->shape().Count(1);
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
    const int64_t feature_grad_rows = feature_grad->shape().At(0);
    const int64_t feature_grad_cols = feature_grad->shape().Count(1);
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
  const int64_t padded_dim =
      std::ceil(static_cast<float>(dim) / static_cast<float>(align_dim)) * align_dim;
  return padded_dim;
}

template<typename T, int32_t N>
struct DotFwdParam {
  const T* in[N];
  int32_t in_feature_dim[N];
  int32_t dim_start_offset[N];
  int32_t features_dim;
  const T* output_concat;
  int32_t output_concat_size;
  T* out;
  int32_t num_in;
};

constexpr int unroll_dim = 2;
template<typename T, typename ComputeType, int32_t N, int32_t pack_size, int TILE_DIM>
__global__ void DotFeatureInteractionTensorCore(
    int M_BLOCKS, int K_BLOCKS, int64_t batch_size, int padded_num_rows, int vector_num_pack,
    int padded_vector_num_pack, int out_num_cols, int out_num_cols_num_pack, int in_shared_mem_cols,
    int in_shared_mem_cols_num_pack, int acc_shared_mem_cols, int acc_shared_mem_cols_num_pack,
    int offset, int output_padding, DotFwdParam<T, N> param) {
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
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (i >= param.num_in) { break; }
      const Pack<T, pack_size>* batch_in = reinterpret_cast<const Pack<T, pack_size>*>(param.in[i])
                                           + batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * unroll_dim; j < param.in_feature_dim[i];
           j += blockDim.y * unroll_dim) {
#pragma unroll
        for (int k = 0; k < unroll_dim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) { break; }
          int buf_row = param.dim_start_offset[i] + in_row;
          buf_pack[buf_row * in_shared_mem_cols_num_pack + col] =
              batch_in[in_row * vector_num_pack + col];
        }
      }
    }
  }
  Pack<T, pack_size> zero;
  for (int k = 0; k < pack_size; ++k) { zero.elem[k] = 0; }
  for (int row = threadIdx.y; row < param.features_dim; row += blockDim.y) {
    for (int col = vector_num_pack + threadIdx.x; col < padded_vector_num_pack; col += blockDim.x) {
      buf_pack[row * in_shared_mem_cols_num_pack + col] = zero;
    }
  }
  __syncthreads();
  for (int blocks_id = warp_id; blocks_id < M_BLOCKS * M_BLOCKS; blocks_id += blockDim.y) {
    int blocks_row_id = blocks_id / M_BLOCKS;
    int blocks_col_id = blocks_id - blocks_row_id * M_BLOCKS;
    if (blocks_row_id >= blocks_col_id) {
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, ComputeType>
          acc;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, T,
                             nvcuda::wmma::row_major>
          a;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, T,
                             nvcuda::wmma::col_major>
          b;
      nvcuda::wmma::fill_fragment(acc, 0.0f);
      for (int step = 0; step < K_BLOCKS; ++step) {
        T* tile_a_ptr = buf + blocks_row_id * TILE_DIM * in_shared_mem_cols + step * TILE_DIM;
        T* tile_b_ptr = buf + blocks_col_id * TILE_DIM * in_shared_mem_cols + step * TILE_DIM;
        nvcuda::wmma::load_matrix_sync(a, tile_a_ptr, in_shared_mem_cols);
        nvcuda::wmma::load_matrix_sync(b, tile_b_ptr, in_shared_mem_cols);
        nvcuda::wmma::mma_sync(acc, a, b, acc);
      }
      ComputeType* tile_ptr =
          acc_buf + blocks_row_id * TILE_DIM * acc_shared_mem_cols + blocks_col_id * TILE_DIM;
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc, acc_shared_mem_cols,
                                      nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  T* emb_out = batch_out + output_concat_size;
  for (int base_row = threadIdx.y * unroll_dim; base_row < param.features_dim;
       base_row += unroll_dim * blockDim.y) {
#pragma unroll
    for (int k = 0; k < unroll_dim; ++k) {
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
}

template<typename T, int N, int32_t pack_size>
struct DotFeatureInteractionKernel {
  static bool Launch(ep::Stream* stream, int64_t batch_size, int concated_padded_dim,
                     int vector_size, int out_num_cols, bool self_interaction, int output_padding,
                     const DotFwdParam<T, N>& param) {
    UNIMPLEMENTED();
    return false;
  }
};

template<int N, int32_t pack_size>
struct DotFeatureInteractionKernel<half, N, pack_size> {
  static bool Launch(ep::Stream* stream, int64_t batch_size, int concated_padded_dim,
                     int vector_size, int out_num_cols, bool self_interaction, int output_padding,
                     const DotFwdParam<half, N>& param) {
    const int block_size = 128;
    const int block_dim_x = 32;
    const int block_dim_y = block_size / block_dim_x;
    const int num_blocks = batch_size;
    const int TILE_DIM = 16;
    const int64_t padded_vector_size = GetPaddedDim(vector_size);
    const int M_BLOCKS = concated_padded_dim / TILE_DIM;
    const int K_BLOCKS = padded_vector_size / TILE_DIM;
    const int skew_half = 8;
    const int skew_acc = 8;  // consider adjust this
    const int in_shared_mem_num_cols = padded_vector_size + skew_half;
    const int acc_shared_mem_num_cols = concated_padded_dim + skew_acc;
    const size_t in_shared_mem_bytes = concated_padded_dim * in_shared_mem_num_cols * sizeof(half);
    using ComputeType = typename DefaultComputeType<half>::type;
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
        DotFeatureInteractionTensorCore<half, ComputeType, N, pack_size, TILE_DIM>, block_size,
        total_shared_mem_bytes));
    if (max_active_blocks <= 0) { return false; }
    cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
    DotFeatureInteractionTensorCore<half, ComputeType, N, pack_size, TILE_DIM>
        <<<num_blocks, dim3(block_dim_x, block_dim_y), total_shared_mem_bytes, cuda_stream>>>(
            M_BLOCKS, K_BLOCKS, batch_size, concated_padded_dim, vector_num_pack,
            padded_vector_num_pack, out_num_cols, out_num_cols_num_pack, in_shared_mem_num_cols,
            in_shared_mem_cols_num_pack, acc_shared_mem_num_cols, acc_shared_mem_cols_num_pack,
            offset, output_padding, param);
    return true;
  }
};

template<typename T, int32_t N>
struct DotBwdParam {
  const T* dy;
  const T* in[N];
  T* in_grad[N];
  T* output_concat_grad;
  int32_t output_concat_size;
  int32_t in_feature_dim[N];
  int32_t dim_start_offset[N];
  int32_t features_dim;
  int32_t num_in;
};

template<typename T, typename ComputeType, int32_t N, int32_t pack_size, int TILE_DIM>
__global__ void DotFeatureInteractionBackwardTensorCore(
    int M_BLOCKS, int N_BLOCKS, int K_BLOCKS, int64_t batch_size, int padded_num_rows,
    int vector_num_pack, int padded_vector_num_pack, int out_num_cols, int out_num_cols_num_pack,
    int in_shared_mem_cols, int in_shared_mem_cols_num_pack, int matrix_dy_shared_mem_cols,
    int matrix_dy_shared_mem_cols_num_pack, int offset, DotBwdParam<T, N> param) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  T* in_buf = reinterpret_cast<T*>(shared_buf);
  Pack<T, pack_size>* in_buf_pack = reinterpret_cast<Pack<T, pack_size>*>(shared_buf);
  T* matrix_dy_buf = in_buf + padded_num_rows * in_shared_mem_cols;
  Pack<T, pack_size>* matrix_dy_pack = reinterpret_cast<Pack<T, pack_size>*>(matrix_dy_buf);
  ComputeType* in_grad_buf =
      reinterpret_cast<ComputeType*>(matrix_dy_buf + padded_num_rows * matrix_dy_shared_mem_cols);
  Pack<ComputeType, pack_size>* in_grad_buf_pack =
      reinterpret_cast<Pack<ComputeType, pack_size>*>(in_grad_buf);

  int batch_idx = blockIdx.x;
  const T* batch_dy = param.dy + batch_idx * out_num_cols;
  const Pack<T, pack_size>* batch_dy_pack =
      reinterpret_cast<const Pack<T, pack_size>*>(param.dy) + batch_idx * out_num_cols_num_pack;
  const int output_concat_size = param.output_concat_size;
  T* batch_output_concat_grad = (param.output_concat_grad)
                                    ? (param.output_concat_grad + batch_idx * output_concat_size)
                                    : nullptr;
  int features_dim = param.features_dim;
  // 1.split dy to concat_out_grad and matrix_dy buf
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  for (int i = thread_id; i < output_concat_size; i += blockDim.x * blockDim.y) {
    batch_output_concat_grad[i] = batch_dy[i];
  }
  const T* batch_interaction_dy = batch_dy + output_concat_size;
  for (int matrix_row = threadIdx.y; matrix_row < padded_num_rows; matrix_row += blockDim.y) {
    for (int matrix_col = threadIdx.x; matrix_col < padded_num_rows; matrix_col += blockDim.x) {
      const int64_t i = matrix_row * matrix_dy_shared_mem_cols + matrix_col;
      T grad_val = 0;
      if (matrix_row < features_dim && matrix_col < features_dim) {
        if (matrix_col < matrix_row) {
          int32_t dy_col_idx = matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = batch_interaction_dy[dy_col_idx];
        } else if (matrix_row < matrix_col) {
          // transpose add
          int32_t trans_row_id = matrix_col;
          int32_t trans_col_id = matrix_row;
          int32_t dy_col_idx =
              trans_row_id * (offset + trans_row_id - 1 + offset) / 2 + trans_col_id;
          grad_val = batch_interaction_dy[dy_col_idx];
        } else if ((matrix_row == matrix_col) && (offset == 1)) {
          int32_t dy_col_idx = matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = batch_interaction_dy[dy_col_idx] * static_cast<T>(2);
        }
      }
      matrix_dy_buf[i] = grad_val;
    }
  }

  // 2.load in to in in_buf
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (i >= param.num_in) { break; }
      const Pack<T, pack_size>* batch_in = reinterpret_cast<const Pack<T, pack_size>*>(param.in[i])
                                           + batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * unroll_dim; j < param.in_feature_dim[i];
           j += blockDim.y * unroll_dim) {
#pragma unroll
        for (int k = 0; k < unroll_dim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) { break; }
          int buf_row = param.dim_start_offset[i] + in_row;
          in_buf_pack[buf_row * in_shared_mem_cols_num_pack + col] =
              batch_in[in_row * vector_num_pack + col];
        }
      }
    }
  }
  Pack<T, pack_size> zero;
  for (int k = 0; k < pack_size; ++k) { zero.elem[k] = 0; }
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

  for (int blocks_id = warp_id; blocks_id < M_BLOCKS * N_BLOCKS; blocks_id += blockDim.y) {
    int blocks_row_id = blocks_id / N_BLOCKS;
    int blocks_col_id = blocks_id - blocks_row_id * N_BLOCKS;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, ComputeType>
        acc;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, T,
                           nvcuda::wmma::row_major>
        a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, T,
                           nvcuda::wmma::row_major>
        b;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    for (int step = 0; step < K_BLOCKS; ++step) {
      // blocks_row_id is a row_id, step is a col_id. blocks_col_id is b col_id,
      // step is b row_id.
      T* tile_a_ptr =
          matrix_dy_buf + blocks_row_id * TILE_DIM * matrix_dy_shared_mem_cols + step * TILE_DIM;
      T* tile_b_ptr = in_buf + step * TILE_DIM * in_shared_mem_cols + blocks_col_id * TILE_DIM;
      nvcuda::wmma::load_matrix_sync(a, tile_a_ptr, matrix_dy_shared_mem_cols);
      nvcuda::wmma::load_matrix_sync(b, tile_b_ptr, in_shared_mem_cols);
      nvcuda::wmma::mma_sync(acc, a, b, acc);
    }
    ComputeType* tile_ptr =
        in_grad_buf + blocks_row_id * TILE_DIM * in_shared_mem_cols + blocks_col_id * TILE_DIM;
    nvcuda::wmma::store_matrix_sync(tile_ptr, acc, in_shared_mem_cols, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();

  // 4.split in_grad buf to dx
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (i >= param.num_in) { break; }
      Pack<T, pack_size>* batch_in_grad = reinterpret_cast<Pack<T, pack_size>*>(param.in_grad[i])
                                          + batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * unroll_dim; j < param.in_feature_dim[i];
           j += blockDim.y * unroll_dim) {
#pragma unroll
        for (int k = 0; k < unroll_dim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) { break; }
          int buf_row = param.dim_start_offset[i] + in_row;
          Pack<T, pack_size> grad_val;
          Pack<ComputeType, pack_size> buf_grad_val =
              in_grad_buf_pack[buf_row * in_shared_mem_cols_num_pack + col];
          for (int t = 0; t < pack_size; ++t) {
            grad_val.elem[t] = static_cast<T>(buf_grad_val.elem[t]);
          }
          batch_in_grad[in_row * vector_num_pack + col] = grad_val;
        }
      }
    }
  }
}

template<typename T, int N, int32_t pack_size>
struct DotFeatureInteractionBackwardKernel {
  static bool Launch(ep::Stream* stream, int64_t batch_size, int concated_padded_dim,
                     int vector_size, int out_num_cols, bool self_interaction,
                     const DotBwdParam<T, N>& param) {
    UNIMPLEMENTED();
    return false;
  }
};

template<int N, int32_t pack_size>
struct DotFeatureInteractionBackwardKernel<half, N, pack_size> {
  static bool Launch(ep::Stream* stream, int64_t batch_size, int concated_padded_dim,
                     int vector_size, int out_num_cols, bool self_interaction,
                     const DotBwdParam<half, N>& param) {
    const int block_size = 256;
    const int block_dim_x = 32;
    const int block_dim_y = block_size / block_dim_x;
    const int num_blocks = batch_size;
    const int TILE_DIM = 16;
    const int64_t padded_vector_size = GetPaddedDim(vector_size);
    const int M_BLOCKS = concated_padded_dim / TILE_DIM;
    const int K_BLOCKS = concated_padded_dim / TILE_DIM;
    const int N_BLOCKS = padded_vector_size / TILE_DIM;
    const int skew_half = 8;
    const int in_shared_mem_num_cols = padded_vector_size + skew_half;
    const int matrix_dy_shared_mem_cols = concated_padded_dim + skew_half;
    const size_t in_shared_mem_bytes = concated_padded_dim * in_shared_mem_num_cols * sizeof(half);
    const size_t matrix_dy_shared_mem_bytes =
        concated_padded_dim * matrix_dy_shared_mem_cols * sizeof(half);
    using ComputeType = typename DefaultComputeType<half>::type;
    const size_t in_grad_shared_mem_bytes =
        concated_padded_dim * in_shared_mem_num_cols * sizeof(ComputeType);
    const size_t total_shared_mem_bytes =
        in_shared_mem_bytes + matrix_dy_shared_mem_bytes + in_grad_shared_mem_bytes;
    const int32_t offset = self_interaction ? 1 : 0;
    const int out_num_cols_num_pack = out_num_cols / pack_size;
    const int vector_num_pack = vector_size / pack_size;
    const int padded_vector_num_pack = padded_vector_size / pack_size;
    const int in_shared_mem_cols_num_pack = in_shared_mem_num_cols / pack_size;
    const int matrix_dy_shared_mem_cols_num_pack = matrix_dy_shared_mem_cols / pack_size;
    int max_active_blocks;
    OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        DotFeatureInteractionBackwardTensorCore<half, ComputeType, N, pack_size, TILE_DIM>,
        block_size, total_shared_mem_bytes));
    if (max_active_blocks <= 0) { return false; }
    cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
    DotFeatureInteractionBackwardTensorCore<half, ComputeType, N, pack_size, TILE_DIM>
        <<<num_blocks, dim3(block_dim_x, block_dim_y), total_shared_mem_bytes, cuda_stream>>>(
            M_BLOCKS, N_BLOCKS, K_BLOCKS, batch_size, concated_padded_dim, vector_num_pack,
            padded_vector_num_pack, out_num_cols, out_num_cols_num_pack, in_shared_mem_num_cols,
            in_shared_mem_cols_num_pack, matrix_dy_shared_mem_cols,
            matrix_dy_shared_mem_cols_num_pack, offset, param);

    return true;
  }
};

template<typename T, int N>
bool DispatchFeatureInteractionDotPackSize(user_op::KernelComputeContext* ctx,
                                           const int32_t input_size) {
  CHECK_LE(input_size, N) << input_size;
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  const int64_t batch_size = out->shape().At(0);
  const int64_t out_num_cols = out->shape().At(1);
  const int64_t vector_size = ctx->TensorDesc4ArgNameAndIndex("features", 0)->shape().At(2);
  DotFwdParam<T, N> param;
  param.num_in = input_size;
  param.out = out->mut_dptr<T>();
  int64_t features_concated_dim = 0;
  for (int i = 0; i < input_size; ++i) {
    param.in[i] = ctx->Tensor4ArgNameAndIndex("features", i)->dptr<T>();
    param.in_feature_dim[i] = ctx->TensorDesc4ArgNameAndIndex("features", i)->shape().At(1);
    param.dim_start_offset[i] = features_concated_dim;
    features_concated_dim += param.in_feature_dim[i];
  }
  const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
  param.features_dim = features_concated_dim;
  if (ctx->has_input("output_concat", 0)) {
    const user_op::Tensor* output_concat = ctx->Tensor4ArgNameAndIndex("output_concat", 0);
    param.output_concat = output_concat->dptr<T>();
    param.output_concat_size = output_concat->shape().At(1);
  } else {
    param.output_concat = nullptr;
    param.output_concat_size = 0;
  }
  const bool self_interaction = ctx->Attr<bool>("self_interaction");
  const int32_t output_padding = ctx->Attr<int32_t>("output_padding");
  if (vector_size % 4 == 0 && out_num_cols % 4 == 0) {
    return DotFeatureInteractionKernel<T, N, 4>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        output_padding, param);
  } else if (vector_size % 2 == 0 && out_num_cols % 2 == 0) {
    return DotFeatureInteractionKernel<T, N, 2>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        output_padding, param);
  } else {
    return DotFeatureInteractionKernel<T, N, 1>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        output_padding, param);
  }
}

template<typename T, int N>
bool DispatchFeatureInteractionDotBackwardPackSize(user_op::KernelComputeContext* ctx,
                                                   const int32_t input_size) {
  CHECK_LE(input_size, N) << input_size;
  user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
  const int64_t batch_size = dy->shape().At(0);
  const int64_t out_num_cols = dy->shape().At(1);
  const int64_t vector_size = ctx->TensorDesc4ArgNameAndIndex("features", 0)->shape().At(2);
  DotBwdParam<T, N> param;
  param.num_in = input_size;
  param.dy = dy->dptr<T>();
  int64_t features_concated_dim = 0;
  for (int i = 0; i < input_size; ++i) {
    param.in[i] = ctx->Tensor4ArgNameAndIndex("features", i)->dptr<T>();
    param.in_grad[i] = ctx->Tensor4ArgNameAndIndex("features_grad", i)->mut_dptr<T>();
    param.in_feature_dim[i] = ctx->TensorDesc4ArgNameAndIndex("features", i)->shape().At(1);
    param.dim_start_offset[i] = features_concated_dim;
    features_concated_dim += param.in_feature_dim[i];
  }
  const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
  param.features_dim = features_concated_dim;
  if (ctx->has_output("output_concat_grad", 0)) {
    user_op::Tensor* output_concat_grad = ctx->Tensor4ArgNameAndIndex("output_concat_grad", 0);
    param.output_concat_grad = output_concat_grad->mut_dptr<T>();
    param.output_concat_size = output_concat_grad->shape().At(1);
  } else {
    param.output_concat_grad = nullptr;
    param.output_concat_size = 0;
  }
  const bool self_interaction = ctx->Attr<bool>("self_interaction");
  if (vector_size % 4 == 0 && out_num_cols % 4 == 0) {
    return DotFeatureInteractionBackwardKernel<T, N, 4>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        param);
  } else if (vector_size % 2 == 0 && out_num_cols % 2 == 0) {
    return DotFeatureInteractionBackwardKernel<T, N, 2>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        param);
  } else {
    return DotFeatureInteractionBackwardKernel<T, N, 1>::Launch(
        ctx->stream(), batch_size, concated_padded_dim, vector_size, out_num_cols, self_interaction,
        param);
  }
}

template<typename T, int32_t N>
struct Param {
  const T* in[N];
  int32_t in_feature_dim[N];
  T* out;
  int32_t num_in;
};

template<typename T, int32_t N, int32_t pack_size>
__global__ void FeatureInteractionSum(int64_t batch_size, int64_t vector_num_pack,
                                      Param<T, N> param) {
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
      for (int i = 0; i < N; ++i) {
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

template<typename T, int32_t N>
struct GradParam {
  const T* dy;
  const T* in[N];
  int32_t in_feature_dim[N];
  T* in_grad[N];
  int32_t num_in;
};

template<typename T, int32_t N>
__global__ void FeatureInteractionSumGrad(int64_t batch_size, int64_t vector_size,
                                          GradParam<T, N> param) {
  using ComputeType = typename DefaultComputeType<T>::type;
  for (int batch_idx = blockIdx.x * blockDim.y + threadIdx.y; batch_idx < batch_size;
       batch_idx += gridDim.x * blockDim.y) {
    const T* batch_dy = param.dy + batch_idx * vector_size;
    for (int col_id = threadIdx.x; col_id < vector_size; col_id += blockDim.x) {
      ComputeType sum = 0;
      for (int i = 0; i < N; ++i) {
        if (i >= param.num_in) { break; }
        const T* batch_in = param.in[i] + batch_idx * param.in_feature_dim[i] * vector_size;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          sum += static_cast<ComputeType>(batch_in[j * vector_size + col_id]);
        }
      }
      for (int i = 0; i < N; ++i) {
        if (i >= param.num_in) { break; }
        const int64_t in_batch_offset = batch_idx * param.in_feature_dim[i] * vector_size;
        const T* batch_in = param.in[i] + in_batch_offset;
        T* batch_in_grad = param.in_grad[i] + in_batch_offset;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          const int64_t offset = j * vector_size + col_id;
          batch_in_grad[offset] =
              static_cast<T>(static_cast<ComputeType>(batch_dy[col_id])
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

template<typename T, int32_t N>
void DispatchFeatureInteractionSumPackSize(ep::Stream* stream, const int64_t batch_size,
                                           const int64_t vector_size, const Param<T, N>& param) {
  int block_dim_x;
  int block_dim_y;
  const int pack_size = (vector_size % 2 == 0) ? 2 : 1;
  const int64_t vector_num_pack = vector_size / pack_size;
  GetBlockDims(vector_num_pack, &block_dim_x, &block_dim_y);
  const int num_blocks = GetNumBlocks(batch_size, block_dim_y);
  dim3 block_dims = dim3(block_dim_x, block_dim_y);
  cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
  if (pack_size == 2) {
    FeatureInteractionSum<T, N, 2>
        <<<num_blocks, block_dims, 0, cuda_stream>>>(batch_size, vector_num_pack, param);
  } else {
    FeatureInteractionSum<T, N, 1>
        <<<num_blocks, block_dims, 0, cuda_stream>>>(batch_size, vector_num_pack, param);
  }
}

template<typename T, int N>
void DispatchFeatureInteractionSumInputSize(user_op::KernelComputeContext* ctx,
                                            const int32_t input_size) {
  CHECK_LE(input_size, N) << input_size;
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  const int64_t batch_size = out->shape().At(0);
  const int64_t vector_size = out->shape().At(1);
  Param<T, N> param;
  param.num_in = input_size;
  param.out = out->mut_dptr<T>();
  for (int i = 0; i < input_size; ++i) {
    param.in[i] = ctx->Tensor4ArgNameAndIndex("features", i)->dptr<T>();
    param.in_feature_dim[i] = ctx->TensorDesc4ArgNameAndIndex("features", i)->shape().At(1);
  }
  DispatchFeatureInteractionSumPackSize<T, N>(ctx->stream(), batch_size, vector_size, param);
}

template<typename T, int N>
void DispatchFeatureInteractionSumGradInputSize(user_op::KernelComputeContext* ctx,
                                                const int32_t input_size) {
  CHECK_LE(input_size, N) << input_size;
  const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
  const int64_t batch_size = dy->shape().At(0);
  const int64_t vector_size = dy->shape().At(1);
  int block_dim_x;
  int block_dim_y;
  GetBlockDims(vector_size, &block_dim_x, &block_dim_y);
  const int num_blocks = GetNumBlocks(batch_size, block_dim_y);
  dim3 block_dims = dim3(block_dim_x, block_dim_y);
  GradParam<T, N> param;
  param.num_in = input_size;
  param.dy = dy->dptr<T>();
  for (int i = 0; i < input_size; ++i) {
    param.in[i] = ctx->Tensor4ArgNameAndIndex("features", i)->dptr<T>();
    param.in_grad[i] = ctx->Tensor4ArgNameAndIndex("features_grad", i)->mut_dptr<T>();
    param.in_feature_dim[i] = ctx->TensorDesc4ArgNameAndIndex("features_grad", i)->shape().At(1);
  }
  FeatureInteractionSumGrad<T, N>
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
    CHECK_LT(out->shape().elem_cnt(), GetMaxVal<int32_t>());
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    if (cuda_stream->device_properties().major >= 7 && data_type == DataType::kFloat16) {
      bool success = TryLaunchTensorCoreDotKernel<T>(ctx);
      if (success == true) { return; }
    }
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t batch_size = out->shape().At(0);
    int64_t features_concated_dim = 0;
    for (int64_t i = 0; i < ctx->input_size("features"); ++i) {
      features_concated_dim += ctx->TensorDesc4ArgNameAndIndex("features", i)->shape().At(1);
    }
    const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
    const int64_t vector_size = ctx->TensorDesc4ArgNameAndIndex("features", 0)->shape().At(2);
    const int64_t out_dim = out->shape().At(1);
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
    CHECK_GE(tmp_buffer->shape().elem_cnt(),
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
      output_concat_end_dim = output_concat->shape().At(1);
      output_concat_ptr = output_concat->dptr<T>();
    }
    CHECK_EQ(valid_out_dim, output_concat_end_dim + interaction_dim);
    GatherConcatKernel<T>(ctx->stream(), out->shape().elem_cnt(), out_dim, valid_out_dim,
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
    if (cuda_stream->device_properties().major >= 7 && data_type == DataType::kFloat16) {
      bool success = TryLaunchTensorCoreDotBackwardKernel<T>(ctx);
      if (success == true) { return; }
    }
    const int64_t batch_size = dy->shape().At(0);
    int64_t features_concated_dim = 0;
    for (int32_t i = 0; i < ctx->output_size("features_grad"); ++i) {
      features_concated_dim += ctx->TensorDesc4ArgNameAndIndex("features_grad", i)->shape().At(1);
    }
    const int64_t concated_padded_dim = GetPaddedDim(features_concated_dim);
    const int64_t vector_size = ctx->TensorDesc4ArgNameAndIndex("features_grad", 0)->shape().At(2);
    const int64_t out_dim = dy->shape().At(1);
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
        tmp_buffer->shape().elem_cnt());
    ConcatFeatures<T>(ctx, batch_size, concated_padded_dim * vector_size,
                      padded_concated_features_ptr);

    T* output_concat_grad_ptr = nullptr;
    int64_t output_concat_end_dim = 0;
    if (ctx->has_output("output_concat_grad", 0)) {
      user_op::Tensor* output_concat_grad = ctx->Tensor4ArgNameAndIndex("output_concat_grad", 0);
      output_concat_grad_ptr = output_concat_grad->mut_dptr<T>();
      output_concat_end_dim = output_concat_grad->shape().At(1);
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
