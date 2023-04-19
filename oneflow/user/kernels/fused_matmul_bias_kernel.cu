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

// same with cublas_fused_mlp_util.cuh
#if CUDA_VERSION >= 11020

namespace oneflow {

namespace {

class FusedMatmulBiasAlgoCache {
  public:
    static FusedMatmulBiasAlgoCache* CreateCache() {
      if (FusedMatmulBiasAlgoCache::cache == nullptr) {
        FusedMatmulBiasAlgoCache::cache = new FusedMatmulBiasAlgoCache();
      }
      return FusedMatmulBiasAlgoCache::cache;
    }
    ~FusedMatmulBiasAlgoCache() = default;
    const cublasLtMatmulAlgo_t* SelectAlgo(const ep::CudaStream* cuda_stream, const CublasFusedMLPKernelCache* matmul_cache, 
      CublasScalarParameter alpha, CublasScalarParameter beta, const user_op::Tensor* weight, 
      const user_op::Tensor* x, const user_op::Tensor* add_to_output, void* y_ptr) {
      auto matmul_desc = matmul_cache->operation_desc;
      auto a_desc = matmul_cache->cublas_a_desc;
      auto b_desc = matmul_cache->cublas_b_desc;
      auto c_desc = matmul_cache->cublas_c_desc;

      int64_t seed = 0;
      std::hash<int64_t> hash_fn;

      HashMatmulDesc_(matmul_desc, &seed, hash_fn);
      HashMatrixLayoutDesc_(a_desc, &seed, hash_fn);
      HashMatrixLayoutDesc_(b_desc, &seed, hash_fn);
      HashMatrixLayoutDesc_(c_desc, &seed, hash_fn);

      auto it = map_.find(seed);
      if (it != map_.end()) {
        return &(it->second.algo);
      }

      int64_t row, col;
      size_t size_to_write;
      cublasLtMatrixLayoutGetAttribute(
          c_desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &row, sizeof(row), &size_to_write);
      cublasLtMatrixLayoutGetAttribute(
          c_desc, CUBLASLT_MATRIX_LAYOUT_COLS, &col, sizeof(col), &size_to_write);

      cublasLtMatmulPreference_t preference = nullptr;
      size_t workspace_size = cuda_stream->cublas_workspace_size();
      OF_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
      OF_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                                                          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_size, sizeof(workspace_size)));
      int returned_results = 0;
      cublasLtMatmulHeuristicResult_t heuristic_result[4];
      OF_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
          cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, matmul_cache->cublas_a_desc,
          matmul_cache->cublas_b_desc, matmul_cache->cublas_c_desc, matmul_cache->cublas_c_desc,
          preference, 4, heuristic_result, &returned_results)); //TODO: magic number 4
      CHECK_GT(returned_results, 0);
      cublasLtMatmulPreferenceDestroy(preference);    

      cudaEvent_t st, ed;
      float ms;
      cudaEventCreate(&st);
      cudaEventCreate(&ed);
      std::vector<std::pair<int, float>> sorted_algos;

      for (int i = 0; i < returned_results; i++) {
        for (int j = 0; j < 128; j++) { //TODO: magic number 128
          OF_CUBLAS_CHECK(cublasLtMatmul(
          cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &alpha, weight->dptr(),
          matmul_cache->cublas_a_desc, x->dptr(), matmul_cache->cublas_b_desc, &beta,
          (add_to_output == nullptr) ? y_ptr : add_to_output->dptr(), matmul_cache->cublas_c_desc,
          y_ptr, matmul_cache->cublas_c_desc, &heuristic_result[i].algo, cuda_stream->cublas_workspace(),
          cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));
        }

        cudaEventRecord(st);
        for (int j = 0; j < 128; j++) { //TODO: magic number 128
          OF_CUBLAS_CHECK(cublasLtMatmul(
          cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &alpha, weight->dptr(),
          matmul_cache->cublas_a_desc, x->dptr(), matmul_cache->cublas_b_desc, &beta,
          (add_to_output == nullptr) ? y_ptr : add_to_output->dptr(), matmul_cache->cublas_c_desc,
          y_ptr, matmul_cache->cublas_c_desc, &heuristic_result[i].algo, cuda_stream->cublas_workspace(),
          cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));
        }
        cudaEventRecord(ed);
        cudaEventSynchronize(ed);
        cudaEventElapsedTime(&ms, st, ed);

        sorted_algos.push_back(std::pair<int, float>(i, ms));
      }

      std::sort(sorted_algos.begin(), sorted_algos.end(), [](auto pair1, auto pair2) {return pair1.second < pair2.second;});

      int fastest_result_id = sorted_algos[0].first;
      map_[seed] = heuristic_result[fastest_result_id];

      return &(map_[seed].algo);
    }
  
  private:
    void HashMatmulDesc_(cublasLtMatmulDesc_t desc,
                       int64_t* seed,
                       const std::hash<int64_t>& hash_fn) {
      size_t size_to_write;
      int trans_a, trans_b;
      uint32_t epilogue;

      cublasLtMatmulDescGetAttribute(desc,
                                      CUBLASLT_MATMUL_DESC_TRANSA,
                                      &trans_a,
                                      sizeof(trans_a),
                                      &size_to_write);
      HashValue_(seed, hash_fn, static_cast<int64_t>(trans_a));

      cublasLtMatmulDescGetAttribute(desc,
                                      CUBLASLT_MATMUL_DESC_TRANSB,
                                      &trans_b,
                                      sizeof(trans_b),
                                      &size_to_write);
      HashValue_(seed, hash_fn, static_cast<int64_t>(trans_b));

      cublasLtMatmulDescGetAttribute(desc,
                                      CUBLASLT_MATMUL_DESC_EPILOGUE,
                                      &epilogue,
                                      sizeof(epilogue),
                                      &size_to_write);
      HashValue_(seed, hash_fn, static_cast<int64_t>(epilogue));
    }

      void HashMatrixLayoutDesc_(cublasLtMatrixLayout_t desc,
                             int64_t* seed,
                             const std::hash<int64_t>& hash_fn) {
      size_t size_to_write;
      uint32_t dtype;
      int32_t batch;
      uint64_t row, col;
      int64_t ld, batch_offset;

      cublasLtMatrixLayoutGetAttribute(desc,
                                        CUBLASLT_MATRIX_LAYOUT_TYPE,
                                        &dtype,
                                        sizeof(dtype),
                                        &size_to_write);
      HashValue_(seed, hash_fn, static_cast<int64_t>(dtype));

      cublasLtMatrixLayoutGetAttribute(
          desc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
          &batch,
          sizeof(batch),
          &size_to_write);
      HashValue_(seed, hash_fn, static_cast<int64_t>(batch));

      cublasLtMatrixLayoutGetAttribute(
          desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &row, sizeof(row), &size_to_write);
      HashValue_(seed, hash_fn, static_cast<int64_t>(row));

      cublasLtMatrixLayoutGetAttribute(
          desc, CUBLASLT_MATRIX_LAYOUT_COLS, &col, sizeof(col), &size_to_write);
      HashValue_(seed, hash_fn, static_cast<int64_t>(col));

      cublasLtMatrixLayoutGetAttribute(
          desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &size_to_write);
      HashValue_(seed, hash_fn, static_cast<int64_t>(ld));

      cublasLtMatrixLayoutGetAttribute(
          desc,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &batch_offset,
          sizeof(batch_offset),
          &size_to_write);
      HashValue_(seed, hash_fn, static_cast<int64_t>(batch_offset));
    }

    void HashValue_(int64_t* seed,
                    const std::hash<int64_t>& hash_fn,
                    int64_t value) {
      *seed ^= hash_fn(value) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
    }

    FusedMatmulBiasAlgoCache() = default;
    std::map<int64_t, cublasLtMatmulHeuristicResult_t> map_;
    inline static FusedMatmulBiasAlgoCache* cache = nullptr;
};

class FusedMatmulBiasKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  FusedMatmulBiasKernel() = default;
  ~FusedMatmulBiasKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const auto* matmul_cache = CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* _add_to_output = (ctx->has_input("_add_to_output", 0))
                                                ? ctx->Tensor4ArgNameAndIndex("_add_to_output", 0)
                                                : nullptr;

    const DataType data_type = out->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    const double alpha = ctx->Attr<double>("alpha");
    const double beta = (ctx->has_input("_add_to_output", 0)) ? ctx->Attr<double>("beta") : 0.0;

    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    DimVector in_shape({x->shape_view().Count(0, x->shape_view().NumAxes() - 1),
                        x->shape_view().At(x->shape_view().NumAxes() - 1)});

    DimVector weight_shape(2);

    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);

    weight->shape_view().ToDimVector(&weight_shape);

    InferMatmulCublasMNK(in_shape, weight_shape,
                         /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                         /*transpose_b=*/ep::primitive::BlasTransposeType::T, &cublas_m, &cublas_n,
                         &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    void* y_ptr = ctx->Tensor4ArgNameAndIndex("out", 0)->mut_dptr();

    SetCublasAttr(matmul_cache, cublas_compute_dtype, cuda_data_type, false,
                  /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                  /*transpose_b=*/ep::primitive::BlasTransposeType::T, epilogue, bias->dptr(),
                  nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc);
    
    FusedMatmulBiasAlgoCache* algo_cache = FusedMatmulBiasAlgoCache::CreateCache();
    const cublasLtMatmulAlgo_t* algo = algo_cache->SelectAlgo(cuda_stream, matmul_cache, sp_alpha, sp_beta, weight, x, _add_to_output, y_ptr);

    OF_CUBLAS_CHECK(cublasLtMatmul(
        cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &sp_alpha, weight->dptr(),
        matmul_cache->cublas_a_desc, x->dptr(), matmul_cache->cublas_b_desc, &sp_beta,
        (_add_to_output == nullptr) ? y_ptr : _add_to_output->dptr(), matmul_cache->cublas_c_desc,
        y_ptr, matmul_cache->cublas_c_desc, algo, cuda_stream->cublas_workspace(),
        cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(data_type)               \
  REGISTER_USER_KERNEL("fused_matmul_bias")                            \
      .SetCreateFn<FusedMatmulBiasKernel>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == data_type));

REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kDouble);
REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kFloat);
REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kFloat16);
#if CUDA_VERSION >= 11000
REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kBFloat16);
#endif  // CUDA_VERSION >= 11000

}  // namespace

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11020
