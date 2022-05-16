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
void ConcatFeatures(user_op::KernelComputeContext* ctx) {
  const int64_t feature_input_size = ctx->input_size("features");
  user_op::Tensor* padded_concated_features =
      ctx->Tensor4ArgNameAndIndex("padded_concated_features", 0);
  auto primitive = ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, 2);
  const int64_t dst_rows = padded_concated_features->shape().At(0);
  const int64_t dst_cols = padded_concated_features->shape().Count(1);
  void* dst_ptr = padded_concated_features->mut_dptr();
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

}  // namespace

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
    const int64_t vector_size = padded_concated_features->shape().At(2);
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

    ConcatFeatures<T>(ctx);
    auto batch_matmul = ep::primitive::NewPrimitive<ep::primitive::BatchMatmulFactory>(
        ctx->device_type(), padded_concated_features->data_type(),
        ep::primitive::BlasTransposeType::N, ep::primitive::BlasTransposeType::T);
    batch_matmul->Launch(ctx->stream(), batch_size, concated_padded_dim, concated_padded_dim,
                         vector_size, 1.0, padded_concated_features->dptr(),
                         padded_concated_features->dptr(), 0.0, matmul_out);

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
    const user_op::TensorDesc& padded_concated_features =
        ctx->InputTensorDesc("padded_concated_features", 0);
    const int64_t batch_size = padded_concated_features.shape().At(0);
    const int64_t vector_size = padded_concated_features.shape().At(2);
    int64_t features_concated_dim = 0;
    for (int32_t i = 0; i < ctx->input_size("features"); ++i) {
      features_concated_dim += ctx->InputTensorDesc("features", i).shape().At(1);
    }
    const int64_t concated_padded_dim = padded_concated_features.shape().At(1);
    const int64_t pad_dim = concated_padded_dim - features_concated_dim;
    size_t pad_tensor_size = GetCudaAlignedSize(batch_size * pad_dim * vector_size * sizeof(T));
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
class FusedDotFeatureInteractionGradKernel final : public user_op::OpKernel,
                                                   public user_op::CudaGraphSupport {
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
    const int64_t vector_size = padded_concated_features->shape().At(2);
    const int64_t out_dim = dy->shape().At(1);
    const bool self_interaction = ctx->Attr<bool>("self_interaction");
    T* matmul_out_grad_ptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>());
    size_t matmul_out_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * concated_padded_dim * sizeof(T));
    T* padded_concated_features_grad_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + matmul_out_grad_size);
    size_t padded_concated_features_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * vector_size * sizeof(T));
    CHECK_LE(matmul_out_grad_size + padded_concated_features_grad_size,
             tmp_buffer->shape().elem_cnt());

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
        ctx->device_type(), padded_concated_features->data_type(),
        ep::primitive::BlasTransposeType::N, ep::primitive::BlasTransposeType::N);
    batch_matmul->Launch(ctx->stream(), batch_size, concated_padded_dim, vector_size,
                         concated_padded_dim, 1.0, matmul_out_grad_ptr,
                         padded_concated_features->dptr(), 0.0, padded_concated_features_grad_ptr);

    ConcatFeaturesGrad(ctx, batch_size, concated_padded_dim, vector_size,
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
    const int64_t vector_size = padded_concated_features_shape.At(2);
    size_t matmul_out_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * concated_padded_dim * sizeof(T));
    size_t padded_concated_features_grad_size =
        GetCudaAlignedSize(batch_size * concated_padded_dim * vector_size * sizeof(T));
    return matmul_out_grad_size + padded_concated_features_grad_size;
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
