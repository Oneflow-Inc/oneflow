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
#include "oneflow/user/kernels/slice_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

inline cublasOperation_t GetCublasOp(char op) {
  switch (op) {
    case 'n':
    case 'N': {
      return CUBLAS_OP_N;
    }
    case 't':
    case 'T': {
      return CUBLAS_OP_T;
    }
    case 'c':
    case 'C': {
      return CUBLAS_OP_C;
    }
    default: {
      UNIMPLEMENTED();
    }
  }
  return CUBLAS_OP_N;
}

template<typename T>
struct CudaDataTypeTrait;

template<>
struct CudaDataTypeTrait<float> {
  const static cudaDataType_t value = CUDA_R_32F;
};

template<>
struct CudaDataTypeTrait<half> {
  const static cudaDataType_t value = CUDA_R_16F;
};

template<typename T>
void CublasBatchGemm(ep::CudaStream* stream, char transa, char transb, int64_t m, int64_t n,
                     int64_t k, T alpha, const T* a, int64_t lda, int64_t stridea, const T* b,
                     int64_t ldb, int64_t strideb, T beta, T* c, int64_t ldc, int64_t stridec,
                     int64_t batch_size) {
  cublasOperation_t opa = GetCublasOp(transa);
  cublasOperation_t opb = GetCublasOp(transb);
  if (CUDA_VERSION >= 9010 && stream->cuda_arch() >= 500) {
#if CUDA_VERSION >= 9010
    cudaDataType_t data_type = CudaDataTypeTrait<T>::value;
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        stream->cublas_handle(), opa, opb, m, n, k, reinterpret_cast<const void*>(&alpha),
        reinterpret_cast<const void*>(a), data_type, lda, stridea, reinterpret_cast<const void*>(b),
        data_type, ldb, strideb, reinterpret_cast<const void*>(&beta), reinterpret_cast<void*>(c),
        data_type, ldc, stridec, batch_size, data_type, CUBLAS_GEMM_DEFAULT));
#else
    UNIMPLEMENTED();
#endif
  }
}

#if CUDA_VERSION >= 9010

template<>
void CublasBatchGemm<half>(ep::CudaStream* stream, char transa, char transb, int64_t m, int64_t n,
                           int64_t k, half alpha, const half* a, int64_t lda, int64_t stridea,
                           const half* b, int64_t ldb, int64_t strideb, half beta, half* c,
                           int64_t ldc, int64_t stridec, int64_t batch_size) {
  using comp_t = float;
  cublasOperation_t opa = GetCublasOp(transa);
  cublasOperation_t opb = GetCublasOp(transb);

  if (stream->cuda_arch() >= 500) {
    float alpha_f = static_cast<comp_t>(alpha);
    float beta_f = static_cast<comp_t>(beta);
#if CUDA_VERSION >= 11000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#endif
    cudaDataType_t data_type = CudaDataTypeTrait<half>::value;
    cudaDataType_t comp_type = CudaDataTypeTrait<comp_t>::value;
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        stream->cublas_handle(), opa, opb, m, n, k, &alpha_f, reinterpret_cast<const void*>(a),
        data_type, lda, stridea, reinterpret_cast<const void*>(b), data_type, ldb, strideb, &beta_f,
        reinterpret_cast<void*>(c), data_type, ldc, stridec, batch_size, comp_type, algo));
  }
}

template<>
void CublasBatchGemm<float16>(ep::CudaStream* stream, char transa, char transb, int64_t m,
                              int64_t n, int64_t k, float16 alpha, const float16* a, int64_t lda,
                              int64_t stridea, const float16* b, int64_t ldb, int64_t strideb,
                              float16 beta, float16* c, int64_t ldc, int64_t stridec,
                              int64_t batch_size) {
  CublasBatchGemm<half>(stream, transa, transb, m, n, k, static_cast<half>(alpha),
                        reinterpret_cast<const half*>(a), lda, stridea,
                        reinterpret_cast<const half*>(b), ldb, strideb, static_cast<half>(beta),
                        reinterpret_cast<half*>(c), ldc, stridec, batch_size);
}

#endif  // CUDA_VERSION >= 9010

template<typename T>
void BatchedGemm(ep::Stream* stream, char opa, char opb, int64_t m, int64_t n, int64_t k,
                 float alpha, const T* a, int64_t lda, int64_t stridea, const T* b, int64_t ldb,
                 int64_t strideb, float beta, T* c, int64_t ldc, int64_t stridec,
                 int64_t batch_size) {
  // swap m and n, a and b to convert from row-major to col-major
  CublasBatchGemm<T>(stream->As<ep::CudaStream>(), opb, opa, n, m, k, static_cast<T>(alpha), b, ldb,
                     strideb, a, lda, stridea, static_cast<T>(beta), c, ldc, stridec, batch_size);
}

SliceParams ConstructSliceParams4Value(int64_t seq_len, int64_t batch_size, int64_t num_heads,
                                       int64_t head_size) {
  // slice (s, b, n, 3, h) to (s, b, n, 1, h)
  SliceParams params;
  params.ndim = 4;
  params.dims[0] = seq_len;
  params.dims[1] = batch_size;
  params.dims[2] = num_heads;
  params.dims[3] = 3 * head_size;
  params.start[0] = 0;
  params.start[1] = 0;
  params.start[2] = 0;
  params.start[3] = 2 * head_size;
  params.step[0] = 1;
  params.step[1] = 1;
  params.step[2] = 1;
  params.step[3] = 1;
  params.size[0] = seq_len;
  params.size[1] = batch_size;
  params.size[2] = num_heads;
  params.size[3] = head_size;
  return params;
}

template<typename T>
void TransposeGpu(ep::Stream* stream, DataType data_type, const ShapeView& in_shape,
                  const ShapeView& out_shape, const std::vector<int32_t>& perm, const T* in,
                  T* out) {
  CHECK_EQ(in_shape.NumAxes(), out_shape.NumAxes());
  int32_t num_axes = in_shape.NumAxes();
  CHECK_EQ(num_axes, perm.size());
  for (int i = 0; i < perm.size(); ++i) { CHECK_EQ(in_shape.At(perm[i]), out_shape.At(i)); }
  auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(stream->device_type(),
                                                                              in_shape.NumAxes());
  CHECK(transpose);
  transpose->Launch(stream, data_type, in_shape.NumAxes(), in_shape.ptr(), in, perm.data(), out);
}

template<typename T>
class FusedSelfAttentionQueryMulKeyAndValueGpuKernel final : public user_op::OpKernel {
 public:
  FusedSelfAttentionQueryMulKeyAndValueGpuKernel() = default;
  ~FusedSelfAttentionQueryMulKeyAndValueGpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* h_tensor = ctx->Tensor4ArgNameAndIndex("hidden_states", 0);
    int64_t seq_len = h_tensor->shape_view().At(0);
    int64_t batch_size = h_tensor->shape_view().At(1);
    int64_t hidden_size = h_tensor->shape_view().At(2);
    int64_t head_size = ctx->Attr<int64_t>("head_size");
    int64_t num_heads = hidden_size / (3 * head_size);
    int64_t ld = batch_size * hidden_size;
    int64_t stride = 3 * head_size;
    int64_t k_offset = head_size;

    // q * k: (sq, b, n, h) x (sk, b, n, h) => (b, n, sq, h) x (b, n, sk, h)
    // => (b, n, sq, h) x (b, n, h, sk) -> (b, n, sq, sk)
    float alpha = ctx->Attr<float>("alpha");
    user_op::Tensor* qmk_tensor = ctx->Tensor4ArgNameAndIndex("query_mul_key", 0);
    const T* q_dptr = h_tensor->dptr<T>();
    const T* k_dptr = h_tensor->dptr<T>() + k_offset;
    BatchedGemm<T>(ctx->stream(), 'N', 'T', seq_len, seq_len, head_size, alpha, q_dptr, ld, stride,
                   k_dptr, ld, stride, 0.0f, qmk_tensor->mut_dptr<T>(), seq_len, seq_len * seq_len,
                   batch_size * num_heads);

    // slice v
    user_op::Tensor* tmp_v_tensor = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* v_tensor = ctx->Tensor4ArgNameAndIndex("value", 0);
    SliceParams params = ConstructSliceParams4Value(seq_len, batch_size, num_heads, head_size);
    SliceKernelUtil<DeviceType::kCUDA, T>::Forward(ctx->stream(), params, h_tensor->dptr<T>(),
                                                   tmp_v_tensor->mut_dptr<T>());
    // v from (s, b, n, h) transpose to (b, n, s, h)
    Shape value_shape({seq_len, batch_size, num_heads, head_size});
    TransposeGpu<T>(ctx->stream(), h_tensor->data_type(), value_shape, v_tensor->shape_view(),
                    {1, 2, 0, 3}, tmp_v_tensor->dptr<T>(), v_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class FusedSelfAttentionQueryMulKeyAndValueGradGpuKernel final : public user_op::OpKernel {
 public:
  FusedSelfAttentionQueryMulKeyAndValueGradGpuKernel() = default;
  ~FusedSelfAttentionQueryMulKeyAndValueGradGpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* v_grad_tensor = ctx->Tensor4ArgNameAndIndex("value_grad", 0);
    const user_op::Tensor* qmk_grad_tensor = ctx->Tensor4ArgNameAndIndex("query_mul_key_grad", 0);
    const user_op::Tensor* h_tensor = ctx->Tensor4ArgNameAndIndex("hidden_states", 0);
    user_op::Tensor* tmp_v_tensor = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* h_grad_tensor = ctx->Tensor4ArgNameAndIndex("hidden_states_grad", 0);

    float alpha = ctx->Attr<float>("alpha");
    int64_t seq_len = h_grad_tensor->shape_view().At(0);
    int64_t batch_size = h_grad_tensor->shape_view().At(1);
    int64_t hidden_size = h_grad_tensor->shape_view().At(2);
    int64_t num_heads = v_grad_tensor->shape_view().At(1);
    int64_t head_size = v_grad_tensor->shape_view().At(3);
    int64_t ld = batch_size * hidden_size;
    int64_t stride = 3 * head_size;
    CHECK_EQ(hidden_size, num_heads * stride);

    // transpose from (b, n, s, h) to (s, b, n, h)
    Shape value_shape({seq_len, batch_size, num_heads, head_size});
    TransposeGpu<T>(ctx->stream(), v_grad_tensor->data_type(), v_grad_tensor->shape_view(),
                    value_shape, {2, 0, 1, 3}, v_grad_tensor->dptr<T>(),
                    tmp_v_tensor->mut_dptr<T>());
    // slice v grad
    SliceParams params = ConstructSliceParams4Value(seq_len, batch_size, num_heads, head_size);
    SliceKernelUtil<DeviceType::kCUDA, T>::Backward(ctx->stream(), params, tmp_v_tensor->dptr<T>(),
                                                    h_grad_tensor->mut_dptr<T>());

    // grad_q = grad_qmk * k
    // (b, n, sq, sk) x (b, n, sk, h) -> (b, n, s, h) <= (s, b, n, h) <= (s, b, n, 3, h)
    const T* qmk_grad_dptr = qmk_grad_tensor->dptr<T>();
    const T* k_dptr = h_tensor->dptr<T>() + head_size;
    T* grad_q_dptr = h_grad_tensor->mut_dptr<T>();
    BatchedGemm<T>(ctx->stream(), 'N', 'N', seq_len, head_size, seq_len, alpha, qmk_grad_dptr,
                   seq_len, seq_len * seq_len, k_dptr, ld, stride, 0.0f, grad_q_dptr, ld, stride,
                   batch_size * num_heads);
    // grad_k = grad_qmk * q
    // (b, n, sk, sq) x (b, n, sq, h) -> (b, n, sk, h) <= (s, b, n, h) <= (s, b, n, 3, h)
    const T* q_dptr = h_tensor->dptr<T>();
    T* grad_k_dptr = h_grad_tensor->mut_dptr<T>() + head_size;
    BatchedGemm<T>(ctx->stream(), 'T', 'N', seq_len, head_size, seq_len, alpha, qmk_grad_dptr,
                   seq_len, seq_len * seq_len, q_dptr, ld, stride, 0.0f, grad_k_dptr, ld, stride,
                   batch_size * num_heads);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t InferTmpBufferSize(user_op::InferContext* ctx) {
  const Shape& value_shape = ctx->OutputShape("value", 0);
  DataType value_dtype = ctx->OutputDType("value", 0);
  return value_shape.elem_cnt() * GetSizeOfDataType(value_dtype);
}

size_t InferGradTmpBufferSize(user_op::InferContext* ctx) {
  const Shape& value_shape = ctx->InputShape("value_grad", 0);
  DataType value_dtype = ctx->InputDType("value_grad", 0);
  return value_shape.elem_cnt() * GetSizeOfDataType(value_dtype);
}

}  // namespace

#define REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_CUDA_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("fused_self_attention_query_mul_key_and_value")                             \
      .SetCreateFn<FusedSelfAttentionQueryMulKeyAndValueGpuKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                             \
                       && (user_op::HobDataType("hidden_states", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTmpBufferSize);

#define REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_GRAD_CUDA_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("fused_self_attention_query_mul_key_and_value_grad")                        \
      .SetCreateFn<FusedSelfAttentionQueryMulKeyAndValueGradGpuKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                             \
                       && (user_op::HobDataType("hidden_states", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferGradTmpBufferSize);

REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_CUDA_KERNEL(float)
REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_CUDA_KERNEL(float16)
REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_GRAD_CUDA_KERNEL(float)
REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_GRAD_CUDA_KERNEL(float16)

}  // namespace oneflow
