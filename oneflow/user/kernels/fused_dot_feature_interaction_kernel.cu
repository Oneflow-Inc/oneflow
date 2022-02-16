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

__global__ void GenerateGatherIndicesGpu(const int32_t concat_dim, const int32_t concat_padded_dim,
                                         const int32_t offset, int32_t* gather_indices) {
  for (int32_t row = threadIdx.y; row < concat_dim; row += blockDim.y) {
    for (int32_t col = threadIdx.x; col < concat_dim; col += blockDim.x) {
      if (col < row + offset) {
        int32_t in_index = row * concat_padded_dim + col;
        int32_t idx = (row - 1 + offset) * (row + offset) / 2 + col;
        gather_indices[idx] = in_index;
      }
    }
  }
}

__global__ void GenerateScatterIndicesGpu(const int32_t concat_dim,
                                          const int32_t concated_padded_dim, const int32_t offset,
                                          int32_t* scatter_indices) {
  for (int32_t row = threadIdx.y; row < concated_padded_dim; row += blockDim.y) {
    for (int32_t col = threadIdx.x; col < concated_padded_dim; col += blockDim.x) {
      int32_t out_idx = row * concated_padded_dim + col;
      if (col < row + offset && row < concat_dim) {
        int32_t in_idx = (row - 1 + offset) * (row + offset) / 2 + col;
        scatter_indices[out_idx] = in_idx;
      } else {
        scatter_indices[out_idx] = -1;
      }
    }
  }
}

template<typename T>
__global__ void GatherConcatGpu(int32_t elem_cnt, int32_t out_dim, int32_t valid_out_dim,
                                int32_t matmul_stride, int32_t output_concat_end_dim,
                                const int32_t* gather_indices, const T* matmul_out,
                                const T* output_concat_ptr, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row = i / out_dim;
    const int32_t col = i - row * out_dim;
    T out_val = 0;
    if (col < output_concat_end_dim) {
      int32_t output_concat_idx = row * output_concat_end_dim + col;
      out_val = output_concat_ptr[output_concat_idx];
    } else if (col < valid_out_dim) {
      int32_t gather_col_idx = gather_indices[col - output_concat_end_dim];
      int32_t gather_idx = row * matmul_stride + gather_col_idx;
      out_val = matmul_out[gather_idx];
    }
    out_ptr[i] = out_val;
  }
}

template<typename T>
__global__ void ScatterSplitGpu(int32_t elem_cnt, int32_t stride_dim, int32_t out_dim,
                                int32_t matmul_stride, int32_t output_concat_end_dim, const T* dy,
                                const int32_t* scatter_indices, T* output_concat_grad,
                                T* matmul_out_grad) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row = i / stride_dim;
    const int32_t col = i - row * stride_dim;
    if (col < output_concat_end_dim) {
      output_concat_grad[row * output_concat_end_dim + col] = dy[row * out_dim + col];
    } else {
      int32_t matmul_col_id = col - output_concat_end_dim;
      int32_t scatter_col_id = scatter_indices[matmul_col_id];
      int32_t matmul_idx = row * matmul_stride + matmul_col_id;
      T grad_val = (scatter_col_id != -1)
                       ? dy[row * out_dim + output_concat_end_dim + scatter_col_id]
                       : static_cast<T>(0);
      matmul_out_grad[matmul_idx] = grad_val;
    }
  }
}

template<typename T>
void ConcatFeatures(user_op::KernelComputeContext* ctx, const T* pad_tensor_ptr,
                    const int64_t pad_dim, const int64_t embedding_size) {
  const int64_t feature_input_size = ctx->input_size("features");
  user_op::Tensor* padded_concated_features =
      ctx->Tensor4ArgNameAndIndex("padded_concated_features", 0);
  auto primitive = ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, 2);
  DimVector dst_shape = {padded_concated_features->shape().At(0),
                         padded_concated_features->shape().Count(1)};
  int64_t out_col_offset = 0;
  for (int64_t i = 0; i < feature_input_size; ++i) {
    const user_op::Tensor* feature = ctx->Tensor4ArgNameAndIndex("features", i);
    const int64_t feature_rows = feature->shape().At(0);
    const int64_t feature_cols = feature->shape().Count(1);
    DimVector dst_pos_vec = {0, out_col_offset};
    DimVector src_shape = {feature_rows, feature_cols};
    DimVector src_pos_vec = {0, 0};
    DimVector extent_vec = {feature_rows, feature_cols};
    primitive->Launch(ctx->stream(), feature->data_type(), 2,
                      padded_concated_features->mut_dptr<T>(), dst_shape.data(), dst_pos_vec.data(),
                      feature->dptr<T>(), src_shape.data(), src_pos_vec.data(), extent_vec.data());
    out_col_offset += feature_cols;
  }
  if (pad_dim > 0) {
    CHECK_EQ(out_col_offset + pad_dim * embedding_size, padded_concated_features->shape().Count(1));
    DimVector dst_pos_vec = {0, out_col_offset};
    DimVector src_shape = {padded_concated_features->shape().At(0), pad_dim * embedding_size};
    DimVector src_pos_vec = {0, 0};
    DimVector extent_vec = {padded_concated_features->shape().At(0), pad_dim * embedding_size};
    primitive->Launch(ctx->stream(), padded_concated_features->data_type(), 2,
                      padded_concated_features->mut_dptr<T>(), dst_shape.data(), dst_pos_vec.data(),
                      pad_tensor_ptr, src_shape.data(), src_pos_vec.data(), extent_vec.data());
  }
}

template<typename T>
void BatchMatmul(ep::Stream* stream, DataType data_type, const bool transpose_b,
                 const int32_t batch_size, const int32_t m, const int32_t n, const int32_t k,
                 const T* in_a, const T* in_b, T* out) {
  float alpha = 1.0f;
  float beta = 0.0f;
  int32_t lda = k;
  int32_t ldb;
  int32_t ldc = n;
  int32_t stride_a = m * k;
  int32_t stride_b = k * n;
  int32_t stride_c = m * n;
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

void GenerateScatterIndices(ep::Stream* stream, const int32_t concat_dim,
                            const int32_t concat_padded_dim, const bool self_interaction,
                            int32_t* scatter_indices) {
  int32_t offset = self_interaction ? 1 : 0;
  GenerateScatterIndicesGpu<<<1, dim3(32, 32), 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
      concat_dim, concat_padded_dim, offset, scatter_indices);
}

void GenerateGatherIndices(ep::Stream* stream, const int32_t concat_dim,
                           const int32_t concat_padded_dim, const bool self_interaction,
                           int32_t* gather_indices) {
  int32_t offset = self_interaction ? 1 : 0;
  GenerateGatherIndicesGpu<<<1, dim3(32, 32), 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
      concat_dim, concat_padded_dim, offset, gather_indices);
}

template<typename T>
void GatherConcatKernel(ep::Stream* stream, int32_t elem_cnt, int32_t out_dim,
                        int32_t interaction_dim, int32_t concated_padded_dim,
                        int32_t output_concat_end_dim, const int32_t* gather_indices,
                        const T* matmul_out, const T* output_concat_ptr, T* out_ptr) {
  int32_t matmul_stride = concated_padded_dim * concated_padded_dim;
  int32_t valid_out_dim = output_concat_end_dim + interaction_dim;
  GatherConcatGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                    stream->As<ep::CudaStream>()->cuda_stream()>>>(
      elem_cnt, out_dim, valid_out_dim, matmul_stride, output_concat_end_dim, gather_indices,
      matmul_out, output_concat_ptr, out_ptr);
}

template<typename T>
void ScatterSplit(ep::Stream* stream, int32_t batch_size, int32_t out_dim,
                  int32_t concated_padded_dim, int32_t output_concat_end_dim, const T* dy,
                  const int32_t* scatter_indices, T* output_concat_grad, T* matmul_out_grad_ptr) {
  int32_t stride_dim = output_concat_end_dim + concated_padded_dim * concated_padded_dim;
  int32_t matmul_stride = concated_padded_dim * concated_padded_dim;
  const int32_t elem_cnt = batch_size * stride_dim;
  ScatterSplitGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                    stream->As<ep::CudaStream>()->cuda_stream()>>>(
      elem_cnt, stride_dim, out_dim, matmul_stride, output_concat_end_dim, dy, scatter_indices,
      output_concat_grad, matmul_out_grad_ptr);
}

template<typename T>
void ConcatFeaturesGrad(user_op::KernelComputeContext* ctx, const int64_t batch_size,
                        const int64_t concated_padded_dim, const int64_t embedding_size,
                        const T* concated_features_grad) {
  auto primitive = ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, 2);
  DimVector src_shape = {batch_size, concated_padded_dim * embedding_size};
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

}  // namespace

template<typename T>
class FusedDotFeatureInteractionKernel final : public user_op::OpKernel {
 public:
  FusedDotFeatureInteractionKernel() = default;
  ~FusedDotFeatureInteractionKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_LT(out->shape().elem_cnt(), GetMaxVal<int32_t>());
    user_op::Tensor* padded_concated_features =
        ctx->Tensor4ArgNameAndIndex("padded_concated_features", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t batch_size = padded_concated_features->shape().At(0);
    int64_t features_concated_dim = 0;
    for (int64_t i = 0; i < ctx->input_size("features"); ++i) {
      features_concated_dim += ctx->TensorDesc4ArgNameAndIndex("features", i)->shape().At(1);
    }
    const int64_t concated_padded_dim = padded_concated_features->shape().At(1);
    const int64_t embedding_size = padded_concated_features->shape().At(2);
    const int64_t out_dim = out->shape().At(1);
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
    GenerateGatherIndices(ctx->stream(), features_concated_dim, concated_padded_dim,
                          self_interaction, gather_indices_ptr);
    T* pad_tensor_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + matmul_out_size + gather_indices_size);
    const int64_t pad_dim = concated_padded_dim - features_concated_dim;
    size_t pad_tensor_size = GetCudaAlignedSize(batch_size * pad_dim * embedding_size * sizeof(T));
    OF_CUDA_CHECK(cudaMemsetAsync(pad_tensor_ptr, 0, pad_tensor_size,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    ConcatFeatures<T>(ctx, pad_tensor_ptr, pad_dim, embedding_size);
    BatchMatmul(ctx->stream(), padded_concated_features->data_type(), true, batch_size,
                concated_padded_dim, concated_padded_dim, embedding_size,
                padded_concated_features->dptr<T>(), padded_concated_features->dptr<T>(),
                matmul_out);
    int64_t output_concat_end_dim = 0;
    const T* output_concat_ptr = nullptr;
    if (ctx->has_input("output_concat", 0)) {
      user_op::Tensor* output_concat = ctx->Tensor4ArgNameAndIndex("output_concat", 0);
      output_concat_end_dim = output_concat->shape().At(1);
      output_concat_ptr = output_concat->dptr<T>();
    }
    GatherConcatKernel<T>(ctx->stream(), out->shape().elem_cnt(), out_dim, interaction_dim,
                          concated_padded_dim, output_concat_end_dim, gather_indices_ptr,
                          matmul_out, output_concat_ptr, out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFusedDotFeatureInteractionInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc& padded_concated_features =
        ctx->InputTensorDesc("padded_concated_features", 0);
    const int64_t batch_size = padded_concated_features.shape().At(0);
    const int64_t embedding_size = padded_concated_features.shape().At(2);
    int64_t features_concated_dim = 0;
    for (int32_t i = 0; i < ctx->input_size("features"); ++i) {
      features_concated_dim += ctx->InputTensorDesc("features", i).shape().At(1);
    }
    const int64_t concated_padded_dim = padded_concated_features.shape().At(1);
    const int64_t pad_dim = concated_padded_dim - features_concated_dim;
    size_t pad_tensor_size = GetCudaAlignedSize(batch_size * pad_dim * embedding_size * sizeof(T));
    size_t matmul_out_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * concated_padded_dim * sizeof(T));
    const bool self_interaction = ctx->Attr<bool>("self_interaction");
    const int64_t interaction_dim = self_interaction
                                        ? features_concated_dim * (features_concated_dim + 1) / 2
                                        : features_concated_dim * (features_concated_dim - 1) / 2;
    size_t gather_indices_size = GetCudaAlignedSize(interaction_dim * sizeof(int32_t));
    return matmul_out_size + gather_indices_size + pad_tensor_size;
  };
}

#define REGISTER_FUSED_DOT_FEATURE_INTERACTION_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("fused_dot_feature_interaction")                                  \
      .SetCreateFn<FusedDotFeatureInteractionKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                   \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(GenFusedDotFeatureInteractionInferTmpSizeFn<dtype>());

REGISTER_FUSED_DOT_FEATURE_INTERACTION_KERNEL(float)
REGISTER_FUSED_DOT_FEATURE_INTERACTION_KERNEL(half)

template<typename T>
class FusedDotFeatureInteractionGradKernel final : public user_op::OpKernel {
 public:
  FusedDotFeatureInteractionGradKernel() = default;
  ~FusedDotFeatureInteractionGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* padded_concated_features =
        ctx->Tensor4ArgNameAndIndex("padded_concated_features", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const DataType data_type = dy->data_type();
    const int64_t batch_size = padded_concated_features->shape().At(0);
    int64_t features_concated_dim = 0;
    for (int32_t i = 0; i < ctx->output_size("features_grad"); ++i) {
      features_concated_dim += ctx->TensorDesc4ArgNameAndIndex("features_grad", i)->shape().At(1);
    }
    const int64_t concated_padded_dim = padded_concated_features->shape().At(1);
    const int64_t embedding_size = padded_concated_features->shape().At(2);
    const int64_t out_dim = dy->shape().At(1);
    const bool self_interaction = ctx->Attr<bool>("self_interaction");
    T* matmul_out_grad_ptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>());
    size_t matmul_out_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * concated_padded_dim * sizeof(T));
    T* transposed_matmul_out_grad_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + matmul_out_grad_size);
    size_t transposed_matmul_out_grad_size = matmul_out_grad_size;
    T* padded_concated_features_grad_ptr = reinterpret_cast<T*>(
        tmp_buffer->mut_dptr<char>() + matmul_out_grad_size + transposed_matmul_out_grad_size);
    size_t padded_concated_features_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * embedding_size * sizeof(T));
    int32_t* scatter_indices_ptr = reinterpret_cast<int32_t*>(
        tmp_buffer->mut_dptr<char>() + matmul_out_grad_size + transposed_matmul_out_grad_size
        + padded_concated_features_grad_size);
    size_t scatter_indices_size =
        GetCudaAlignedSize(concated_padded_dim * concated_padded_dim * sizeof(int32_t));
    GenerateScatterIndices(ctx->stream(), features_concated_dim, concated_padded_dim,
                           self_interaction, scatter_indices_ptr);

    T* output_concat_grad_ptr = nullptr;
    int64_t output_concat_end_dim = 0;
    if (ctx->has_output("output_concat_grad", 0)) {
      user_op::Tensor* output_concat_grad = ctx->Tensor4ArgNameAndIndex("output_concat_grad", 0);
      output_concat_grad_ptr = output_concat_grad->mut_dptr<T>();
      output_concat_end_dim = output_concat_grad->shape().At(1);
    }
    ScatterSplit(ctx->stream(), batch_size, out_dim, concated_padded_dim, output_concat_end_dim,
                 dy->dptr<T>(), scatter_indices_ptr, output_concat_grad_ptr, matmul_out_grad_ptr);
    const int32_t num_dims = 3;
    DimVector transpose_dims = {batch_size, concated_padded_dim, concated_padded_dim};
    std::vector<int32_t> perm = {0, 2, 1};
    const int64_t count = batch_size * concated_padded_dim * concated_padded_dim;
    auto transpose_primitive =
        ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(DeviceType::kCUDA, 3);
    transpose_primitive->Launch(ctx->stream(), data_type, num_dims, transpose_dims.data(),
                                matmul_out_grad_ptr, perm.data(), transposed_matmul_out_grad_ptr);

    auto add_primitive =
        ep::primitive::NewPrimitive<ep::primitive::AddFactory>(DeviceType::kCUDA, data_type);
    add_primitive->Launch(ctx->stream(), matmul_out_grad_ptr, transposed_matmul_out_grad_ptr,
                          matmul_out_grad_ptr, count);

    BatchMatmul(ctx->stream(), data_type, false, batch_size, concated_padded_dim, embedding_size,
                concated_padded_dim, matmul_out_grad_ptr, padded_concated_features->dptr<T>(),
                padded_concated_features_grad_ptr);

    ConcatFeaturesGrad(ctx, batch_size, concated_padded_dim, embedding_size,
                       padded_concated_features_grad_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFusedDotFeatureInteractionGradInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const auto& padded_concated_features_shape =
        ctx->InputTensorDesc("padded_concated_features", 0).shape();
    const int64_t batch_size = padded_concated_features_shape.At(0);
    const int64_t concated_padded_dim = padded_concated_features_shape.At(1);
    const int64_t embedding_size = padded_concated_features_shape.At(2);
    size_t matmul_out_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * concated_padded_dim * sizeof(T));
    size_t transposed_matmul_out_grad_size = matmul_out_grad_size;
    size_t padded_concated_features_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * embedding_size * sizeof(T));
    size_t scatter_indices_size =
        GetCudaAlignedSize(concated_padded_dim * concated_padded_dim * sizeof(int32_t));
    return matmul_out_grad_size + transposed_matmul_out_grad_size
           + padded_concated_features_grad_size + scatter_indices_size;
  };
}

#define REGISTER_FUSED_DOT_FEATURE_INTERACTION_GRAD_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("fused_dot_feature_interaction_grad")                            \
      .SetCreateFn<FusedDotFeatureInteractionGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(GenFusedDotFeatureInteractionGradInferTmpSizeFn<dtype>());

REGISTER_FUSED_DOT_FEATURE_INTERACTION_GRAD_KERNEL(float)
REGISTER_FUSED_DOT_FEATURE_INTERACTION_GRAD_KERNEL(half)

}  // namespace oneflow
