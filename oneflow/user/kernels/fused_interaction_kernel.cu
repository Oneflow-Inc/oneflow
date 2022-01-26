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
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/include/primitive/add.h"

namespace oneflow {

namespace {

void DumpToFile(ep::Stream* stream, std::string filename, int64_t parallel_id, size_t data_size,
                const void* ptr) {
  void* host_ptr;
  OF_CUDA_CHECK(cudaMallocHost(&host_ptr, data_size));
  std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                ep::primitive::MemcpyKind::kDtoH);
  CHECK(copyd2h_primitive);
  copyd2h_primitive->Launch(stream, host_ptr, ptr, data_size);
  CHECK_JUST(stream->Sync());
  std::ofstream dx_os;
  dx_os.open(StrCat("test/" + filename + "_", parallel_id));
  dx_os.write(reinterpret_cast<char*>(host_ptr), data_size);
  dx_os.close();
  OF_CUDA_CHECK(cudaFreeHost(host_ptr));
}

}  // namespace

template<typename T>
void ConcatKernel(ep::Stream* stream, DataType data_type, const int64_t rows,
                  const int64_t out_cols, const std::vector<int64_t>& in_cols,
                  const std::vector<const void*>& in_ptrs, void* out_ptr) {
  auto primitive = ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, 2);
  int64_t out_col_offset = 0;
  for (int i = 0; i < in_cols.size(); ++i) {
    if (in_cols.at(i) > 0) {
      DimVector dst_shape = {rows, out_cols};
      DimVector dst_pos_vec = {0, out_col_offset};
      DimVector src_shape = {rows, in_cols.at(i)};
      DimVector src_pos_vec = {0, 0};
      DimVector extent_vec = {rows, in_cols.at(i)};
      primitive->Launch(stream, data_type, 2, out_ptr, dst_shape.data(), dst_pos_vec.data(),
                        in_ptrs.at(i), src_shape.data(), src_pos_vec.data(), extent_vec.data());
    }
    out_col_offset += in_cols.at(i);
  }
}

template<typename T>
void BatchMatmul(ep::Stream* stream, DataType data_type, const bool transpose_b,
                 const int64_t batch_size, const int64_t m, const int64_t n, const int64_t k,
                 const T* in_a, const T* in_b, T* out) {
  float alpha = 1.0f;
  float beta = 0.0f;
  int lda = k;
  int ldb;
  int ldc = n;
  int stride_a = m * k;
  int stride_b = k * n;
  int stride_c = m * n;
  cublasOperation_t trans_b{};
  if (transpose_b) {
    trans_b = CUBLAS_OP_T;
    ldb = k;
  } else {
    trans_b = CUBLAS_OP_N;
    ldb = n;
  }
#if CUDA_VERSION >= 11000
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
  cublasGemmAlgo_t algo =
      (data_type == DataType::kFloat16) ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
#endif
  cudaDataType_t cuda_data_type;
  cudaDataType_t compute_type = CUDA_R_32F;
  if (data_type == DataType::kFloat16) {
    cuda_data_type = CUDA_R_16F;
  } else if (data_type == DataType::kFloat) {
    cuda_data_type = CUDA_R_32F;
  } else {
    UNIMPLEMENTED();
  }
  OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      stream->As<ep::CudaStream>()->cublas_handle(), trans_b, CUBLAS_OP_N, n, m, k, &alpha, in_b,
      cuda_data_type, ldb, stride_b, in_a, cuda_data_type, lda, stride_a, &beta, out,
      cuda_data_type, ldc, stride_c, batch_size, compute_type, algo));
}

template<typename T>
__global__ void gather_concat_bprop_kernel(const T* out, T* in0, T* mat, const int h,
                                           const int n_pad, const int n_ins, const int w) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* s_buf = reinterpret_cast<T*>(shared_buf);
  for (int bid = blockIdx.x; bid < h; bid += gridDim.x) {
    int tid_base = threadIdx.y * blockDim.x + threadIdx.x;
    int out_len = w + (n_ins * (n_ins + 1) / 2 - n_ins) + 1;
    int g_out_idx_base = bid * out_len;
    for (int tid = tid_base; tid < out_len - 1; tid += blockDim.y * blockDim.x) {
      int g_out_idx = g_out_idx_base + tid;
      T val = out[g_out_idx];
      if (tid < w) {
        in0[bid * w + tid] = val;
      } else {
        s_buf[tid - w] = val;
      }
    }
    __syncthreads();

    int g_in_idx_base = bid * n_pad * n_pad;
    for (int row = threadIdx.y; row < n_ins; row += blockDim.y) {
      for (int col = threadIdx.x; col < n_ins; col += blockDim.x) {
        int idx_in_blk = row * n_pad + col;
        int g_in_idx = g_in_idx_base + idx_in_blk;
        int s_idx = (col * (col - 1) / 2) + row;
        mat[g_in_idx] = (col > row) ? s_buf[s_idx] : T(0);
      }
    }
    __syncthreads();
  }
}

__global__ void GenerateTrilIndicesGpu(const int32_t concat_dim, const int32_t pad_dim,
                                       int32_t* tril_indices) {
  for (int row = threadIdx.y; row < concat_dim; row += blockDim.y) {
    for (int col = threadIdx.x; col < concat_dim; col += blockDim.x) {
      if (col > row) {
        int in_index = row * pad_dim + col;
        int idx = (col * (col - 1) / 2) + row;
        tril_indices[idx] = in_index;
      }
    }
  }
}

template<typename T>
__global__ void GatherConcatGpu(int64_t batch_size, int64_t out_dim, int64_t tril_dim,
                                int64_t pad_dim, int64_t embedding_size,
                                const int32_t* tril_indices, const T* matmul_out,
                                const T* dense_feature_ptr, T* out_ptr) {
  for (int row = blockIdx.x; row < batch_size; row += gridDim.x) {
    const T* row_matmul = matmul_out + row * pad_dim * pad_dim;
    const T* row_dense_feature = dense_feature_ptr + row * embedding_size;
    T* row_out = out_ptr + row * out_dim;
    for (int col = threadIdx.x; col < out_dim; col += blockDim.x) {
      T out_val = 0;
      if (col < embedding_size) {
        out_val = row_dense_feature[col];
      } else if (col < embedding_size + tril_dim) {
        int32_t index = tril_indices[col - embedding_size];
        out_val = row_matmul[index];
      }
      row_out[col] = out_val;
    }
  }
}

template<typename T>
__global__ void ScatterSplitGpu(int64_t batch_size, int64_t out_dim, int64_t pad_dim,
                                int64_t embedding_size, const T* dy, T* dense_feature_grad,
                                T* matmul_out_grad) {
  for (int row = blockIdx.x; row < batch_size; row += gridDim.x) {
    const T* row_dy = dy + row * out_dim;
    T* row_dense_feature_grad = dense_feature_grad + row * embedding_size;
    T* row_matmul_out_grad = matmul_out_grad + row * pad_dim * pad_dim;
    for (int col = threadIdx.x; col < embedding_size + pad_dim * pad_dim; col += blockDim.x) {
      if (col < embedding_size) {
        row_dense_feature_grad[col] = row_dy[col];
      } else {
        int sparse_col_id = col - embedding_size;
        int i = sparse_col_id / pad_dim;
        int j = sparse_col_id - i * pad_dim;
        T sparse_grad = 0;
        if (j > i) {
          int dy_idx = (j * (j - 1) / 2) + i;
          sparse_grad = row_dy[embedding_size + dy_idx];
        }
        row_matmul_out_grad[sparse_col_id] = sparse_grad;
      }
    }
  }
}

void GenerateTrilIndices(ep::Stream* stream, const int32_t concat_dim, const int32_t pad_dim,
                         int32_t* tril_indices) {
  GenerateTrilIndicesGpu<<<1, (32, 32), 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
      concat_dim, pad_dim, tril_indices);
}

// out_dim 480 tril_dim 351 pad_dim 32 concat_dim 27
template<typename T>
void GatherConcatKernel(ep::Stream* stream, int64_t batch_size, int64_t out_dim, int64_t tril_dim,
                        int64_t pad_dim, int64_t concat_dim, int64_t embedding_size,
                        const int32_t* tril_indices, const T* matmul_out,
                        const T* dense_feature_ptr, T* out_ptr) {
  GatherConcatGpu<<<batch_size, 256, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
      batch_size, out_dim, tril_dim, pad_dim, embedding_size, tril_indices, matmul_out,
      dense_feature_ptr, out_ptr);
}

template<typename T>
void ScatterSplitKernel(ep::Stream* stream, int64_t batch_size, int64_t out_dim, int64_t pad_dim,
                        int64_t concat_dim, int64_t embedding_size, const T* dy,
                        T* dense_feature_grad, T* matmul_out_grad_ptr) {
  ScatterSplitGpu<<<batch_size, 256, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
      batch_size, out_dim, pad_dim, embedding_size, dy, dense_feature_grad, matmul_out_grad_ptr);
}

template<typename T>
__global__ void SliceAddGpu(const int64_t batch_size, const int64_t pad_dim,
                            const int64_t embedding_size, const T* concat_out_grad_ptr,
                            T* dense_feature_grad) {
  for (int row = blockIdx.x; row < batch_size; row += gridDim.x) {
    for (int col = threadIdx.x; col < embedding_size; col += blockDim.x) {
      const int64_t out_offset = row * embedding_size + col;
      const int64_t in_offset = row * pad_dim * embedding_size + col;
      dense_feature_grad[out_offset] += concat_out_grad_ptr[in_offset];
    }
  }
}

template<typename T>
void SplitAddKernel(ep::Stream* stream, DataType data_type, int64_t batch_size, int64_t pad_dim,
                    int64_t concat_dim, int64_t embedding_size, const T* concat_out_grad_ptr,
                    T* dense_feature_grad, T* sparse_feature_grad) {
  // dense feature grad
  SliceAddGpu<T><<<batch_size, embedding_size, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
      batch_size, pad_dim, embedding_size, concat_out_grad_ptr, dense_feature_grad);

  // sparse feature grad
  auto primitive = ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, 2);
  DimVector dst_shape = {batch_size, (concat_dim - 1) * embedding_size};
  DimVector dst_pos_vec = {0, 0};
  DimVector src_shape = {batch_size, pad_dim * embedding_size};
  DimVector src_pos_vec = {0, 1 * embedding_size};
  DimVector extent_vec = {batch_size, (concat_dim - 1) * embedding_size};
  primitive->Launch(stream, data_type, 2, sparse_feature_grad, dst_shape.data(), dst_pos_vec.data(),
                    concat_out_grad_ptr, src_shape.data(), src_pos_vec.data(), extent_vec.data());
}

int64_t GetPadDim(const int64_t dim) { return std::ceil(static_cast<float>(dim) / 8) * 8; }

template<typename T>
class FusedInteractionKernel final : public user_op::OpKernel {
 public:
  FusedInteractionKernel() = default;
  ~FusedInteractionKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dense_feature = ctx->Tensor4ArgNameAndIndex("dense_feature", 0);
    const user_op::Tensor* sparse_feature = ctx->Tensor4ArgNameAndIndex("sparse_feature", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* concat_out = ctx->Tensor4ArgNameAndIndex("concat_out", 0)->mut_dptr<T>();
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    CHECK_EQ(dense_feature->shape().NumAxes(), 2);
    CHECK_EQ(sparse_feature->shape().NumAxes(), 3);
    const int64_t batch_size = dense_feature->shape().At(0);
    const int64_t num_columns = sparse_feature->shape().At(1);
    const int64_t concat_dim = num_columns + 1;
    const int64_t embedding_size = dense_feature->shape().At(1);

    const int64_t pad_dim = GetPadDim(concat_dim);
    T* pad_tensor_ptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>());
    size_t pad_tensor_size =
        GetCudaAlignedSize(batch_size * (pad_dim - concat_dim) * embedding_size * sizeof(T));
    OF_CUDA_CHECK(cudaMemsetAsync(pad_tensor_ptr, 0, pad_tensor_size,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    // T* concat_out = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + pad_tensor_size);
    size_t concat_out_size = GetCudaAlignedSize(batch_size * pad_dim * embedding_size * sizeof(T));
    T* matmul_out =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + pad_tensor_size + concat_out_size);
    size_t matmul_out_size = GetCudaAlignedSize(batch_size * pad_dim * pad_dim * sizeof(T));

    const int64_t tril_dim = num_columns * (num_columns + 1) / 2;
    const int64_t out_dim = out->shape().At(1);
    int32_t* tril_indices_ptr = reinterpret_cast<int32_t*>(tmp_buffer->mut_dptr<char>()
                                                           + pad_tensor_size + matmul_out_size);
    size_t tril_indices_size = GetCudaAlignedSize(tril_dim * sizeof(int32_t));
    GenerateTrilIndices(ctx->stream(), concat_dim, pad_dim, tril_indices_ptr);

    std::vector<int64_t> in_cols;
    in_cols.push_back(1 * embedding_size);
    in_cols.push_back(sparse_feature->shape().At(1) * embedding_size);
    in_cols.push_back((pad_dim - concat_dim) * embedding_size);
    std::vector<const void*> in_ptrs;
    in_ptrs.push_back(dense_feature->dptr());
    in_ptrs.push_back(sparse_feature->dptr());
    in_ptrs.push_back(pad_tensor_ptr);
    // bsz, 32, 128
    ConcatKernel<T>(ctx->stream(), dense_feature->data_type(), batch_size, pad_dim * embedding_size,
                    in_cols, in_ptrs, reinterpret_cast<void*>(concat_out));
    // bsz, 32, 32
    BatchMatmul(ctx->stream(), dense_feature->data_type(), true, batch_size, pad_dim, pad_dim,
                embedding_size, concat_out, concat_out, matmul_out);

    GatherConcatKernel<T>(ctx->stream(), batch_size, out_dim, tril_dim, pad_dim, concat_dim,
                          embedding_size, tril_indices_ptr, matmul_out, dense_feature->dptr<T>(),
                          out->mut_dptr<T>());

    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFusedInteractionInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc& sparse_feature = ctx->InputTensorDesc("sparse_feature", 0);
    const int64_t batch_size = sparse_feature.shape().At(0);
    const int64_t num_columns = sparse_feature.shape().At(1);
    const int64_t concat_dim = num_columns + 1;
    const int64_t embedding_size = sparse_feature.shape().At(2);
    const int64_t pad_dim = GetPadDim(concat_dim);
    size_t pad_tensor_size =
        GetCudaAlignedSize(batch_size * (pad_dim - concat_dim) * embedding_size * sizeof(T));
    size_t matmul_out_size = GetCudaAlignedSize(batch_size * pad_dim * pad_dim * sizeof(T));
    size_t tril_indices_size =
        GetCudaAlignedSize(num_columns * (num_columns + 1) / 2 * sizeof(int32_t));
    return pad_tensor_size + matmul_out_size + tril_indices_size;
  };
}

#define REGISTER_FUSED_INTERACTION_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("fused_interaction")                                                        \
      .SetCreateFn<FusedInteractionKernel<dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                             \
                       && (user_op::HobDataType("dense_feature", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(GenFusedInteractionInferTmpSizeFn<dtype>());

REGISTER_FUSED_INTERACTION_KERNEL(float)
REGISTER_FUSED_INTERACTION_KERNEL(half)

template<typename T>
class FusedInteractionGradKernel final : public user_op::OpKernel {
 public:
  FusedInteractionGradKernel() = default;
  ~FusedInteractionGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* concat_out = ctx->Tensor4ArgNameAndIndex("concat_out", 0);
    user_op::Tensor* dense_feature_grad = ctx->Tensor4ArgNameAndIndex("dense_feature_grad", 0);
    user_op::Tensor* sparse_feature_grad = ctx->Tensor4ArgNameAndIndex("sparse_feature_grad", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t batch_size = dense_feature_grad->shape().At(0);
    const int64_t concat_dim = sparse_feature_grad->shape().At(1) + 1;
    const int64_t embedding_size = dense_feature_grad->shape().At(1);
    const int64_t pad_dim = GetPadDim(concat_dim);
    const int64_t out_dim = dy->shape().At(1);
    T* matmul_out_grad_ptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>());
    size_t matmul_out_grad_size = GetCudaAlignedSize(batch_size * pad_dim * pad_dim * sizeof(T));
    T* transposed_matmul_out_grad_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + matmul_out_grad_size);
    size_t transposed_matmul_out_grad_size = matmul_out_grad_size;
    T* concat_out_grad_ptr = reinterpret_cast<T*>(
        tmp_buffer->mut_dptr<char>() + matmul_out_grad_size + transposed_matmul_out_grad_size);
    size_t concat_out_grad_size =
        GetCudaAlignedSize(batch_size * pad_dim * embedding_size * sizeof(T));

    ScatterSplitKernel(ctx->stream(), batch_size, out_dim, pad_dim, concat_dim, embedding_size,
                       dy->dptr<T>(), dense_feature_grad->mut_dptr<T>(), matmul_out_grad_ptr);
    const int64_t num_dims = 3;
    DimVector transpose_dims = {batch_size, pad_dim, pad_dim};
    std::vector<int32_t> perm = {0, 2, 1};
    const int64_t count = batch_size * pad_dim * pad_dim;
    auto transpose_primitive =
        ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(DeviceType::kCUDA, 3);
    transpose_primitive->Launch(ctx->stream(), dy->data_type(), num_dims, transpose_dims.data(),
                                matmul_out_grad_ptr, perm.data(), transposed_matmul_out_grad_ptr);

    auto add_primitive =
        ep::primitive::NewPrimitive<ep::primitive::AddFactory>(DeviceType::kCUDA, dy->data_type());
    add_primitive->Launch(ctx->stream(), matmul_out_grad_ptr, transposed_matmul_out_grad_ptr,
                          matmul_out_grad_ptr, count);

    BatchMatmul(ctx->stream(), dense_feature_grad->data_type(), false, batch_size, pad_dim,
                embedding_size, pad_dim, matmul_out_grad_ptr, concat_out->dptr<T>(),
                concat_out_grad_ptr);

    SplitAddKernel<T>(ctx->stream(), dense_feature_grad->data_type(), batch_size, pad_dim,
                      concat_dim, embedding_size, concat_out_grad_ptr,
                      dense_feature_grad->mut_dptr<T>(), sparse_feature_grad->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFusedInteractionGradInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc& sparse_feature_grad = ctx->InputTensorDesc("sparse_feature_grad", 0);
    const int64_t batch_size = sparse_feature_grad.shape().At(0);
    const int64_t concat_dim = sparse_feature_grad.shape().At(1) + 1;
    const int64_t embedding_size = sparse_feature_grad.shape().At(2);
    const int64_t pad_dim = GetPadDim(concat_dim);
    size_t matmul_out_grad_size = GetCudaAlignedSize(batch_size * pad_dim * pad_dim * sizeof(T));
    size_t transposed_matmul_out_grad_size = matmul_out_grad_size;
    size_t concat_out_grad_size =
        GetCudaAlignedSize(batch_size * pad_dim * embedding_size * sizeof(T));

    return matmul_out_grad_size + transposed_matmul_out_grad_size + concat_out_grad_size;
  };
}

#define REGISTER_FUSED_INTERACTION_GRAD_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("fused_interaction_grad")                                        \
      .SetCreateFn<FusedInteractionGradKernel<dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(GenFusedInteractionGradInferTmpSizeFn<dtype>());

REGISTER_FUSED_INTERACTION_GRAD_KERNEL(float)
REGISTER_FUSED_INTERACTION_GRAD_KERNEL(half)

}  // namespace oneflow
