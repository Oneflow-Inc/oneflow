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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/cublas_fused_mlp_util.cuh"
// CUBLAS_AUX_EPILOGUE only support in cuda11.4 or higher version, in cuda11.4 it need static link.
#if CUDA_VERSION >= 11060

namespace oneflow {

namespace {

template<typename T>
class CublasFusedMLPKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  CublasFusedMLPKernel() = default;
  ~CublasFusedMLPKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    /*
    Fused DenseActivation Layer. Assume we have two layers:
    A: (m, k)
    B: (n, k) need transpose
    C: (j, n) need transpose
    tmp: A matmul B(transpose), its shape is (m, n)
    out: tmp matmul C(transpose), its shape is (m, j)
    */
    const int32_t weight_size = ctx->input_size("weights");
    const int32_t bias_size = ctx->input_size("biases");
    CHECK_EQ(weight_size, bias_size) << "The number of weight and bias is not equal!. ";
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const auto* matmul_cache = CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool skip_final_activation = ctx->Attr<bool>("skip_final_activation");

    const DataType data_type = out->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    const double alpha = 1.0;
    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    const double beta = 0.0;
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    // Currently only support 2D matmul.
    DimVector in_shape(2);
    x->shape_view().ToDimVector(&in_shape);

    DimVector weight_shape(2);

    const void* in_buf_ptr = x->dptr();
    for (int idx = 0; idx < weight_size; idx++) {
      const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weights", idx);
      const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("biases", idx);
      user_op::Tensor* cublas_aux = ctx->Tensor4ArgNameAndIndex("cublas_aux", idx);

      int64_t out_feature = weight->shape_view().At(0);
      weight->shape_view().ToDimVector(&weight_shape);

      InferMatmulCublasMNK(in_shape, weight_shape,
                           /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                           /*transpose_b=*/ep::primitive::BlasTransposeType::T, &cublas_m,
                           &cublas_n, &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

      cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_AUX_BIAS;
      bool need_aux = true;
      void* y_ptr = nullptr;

      if (idx == weight_size - 1) {
        y_ptr = ctx->Tensor4ArgNameAndIndex("out", 0)->mut_dptr();
        if (skip_final_activation) {
          epilogue = CUBLASLT_EPILOGUE_BIAS;
          need_aux = false;
        }
      } else {
        y_ptr = ctx->Tensor4ArgNameAndIndex("hidden", idx)->mut_dptr();
      }
      SetCublasAttr(matmul_cache, cublas_compute_dtype, cuda_data_type, need_aux,
                    /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                    /*transpose_b=*/ep::primitive::BlasTransposeType::T, epilogue, bias->dptr(),
                    cublas_aux->dptr(), cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb,
                    cublas_ldc);

      OF_CUBLAS_CHECK(cublasLtMatmul(
          cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &sp_alpha, weight->dptr(),
          matmul_cache->cublas_a_desc, in_buf_ptr, matmul_cache->cublas_b_desc, &sp_beta, y_ptr,
          matmul_cache->cublas_c_desc, y_ptr, matmul_cache->cublas_c_desc, nullptr,
          cuda_stream->cublas_workspace(), cuda_stream->cublas_workspace_size(),
          cuda_stream->cuda_stream()));

      // Set hidden_layer ptr as next layer's input.
      in_buf_ptr = y_ptr;
      // Set hidden_layer shape as next layer's input shape.
      in_shape.at(1) = out_feature;
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUBLAS_FUSED_MLP_KERNEL_GPU(cpp_type, data_type)      \
  REGISTER_USER_KERNEL("cublas_fused_mlp")                             \
      .SetCreateFn<CublasFusedMLPKernel<cpp_type>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == data_type));

REGISTER_CUBLAS_FUSED_MLP_KERNEL_GPU(double, DataType::kDouble);
REGISTER_CUBLAS_FUSED_MLP_KERNEL_GPU(float, DataType::kFloat);
REGISTER_CUBLAS_FUSED_MLP_KERNEL_GPU(half, DataType::kFloat16);
REGISTER_CUBLAS_FUSED_MLP_KERNEL_GPU(nv_bfloat16, DataType::kBFloat16);

}  // namespace

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11060
