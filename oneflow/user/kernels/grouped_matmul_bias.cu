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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

struct Problem {
  Problem(int64_t m, int64_t n, int64_t k) : m(m), n(n), k(k) {}
  int64_t m;
  int64_t n;
  int64_t k;
};

inline bool operator==(const Problem& lhs, const Problem& rhs) {
  return lhs.m == rhs.m && lhs.n == rhs.n && lhs.k == rhs.k;
}

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::Problem> {
  std::size_t operator()(const oneflow::Problem& p) const {
    return oneflow::Hash<int64_t, int64_t, int64_t>(p.m, p.n, p.k);
  }
};

}  // namespace std

namespace oneflow {

namespace {

constexpr int64_t kMaxProblemBatch = 64;

template<typename T>
struct Buffer {
  const T* x;
  const T* w;
  const T* b;
  T* y;
};

template<typename T>
struct Param {
  Param(const Problem& problem, std::vector<Buffer<T>> buffers)
      : problem(problem), n(buffers.size()) {
    std::copy(buffers.cbegin(), buffers.cend(), buffer);
    elem_cnt = n * problem.m * problem.n;
  }
  Problem problem;
  Buffer<T> buffer[kMaxProblemBatch];
  int n;
  int elem_cnt;
};

template<typename T, bool has_biases>
__global__ void InitPtrAndApplyBias(Param<T> p, void** ptr_arr) {
  if (has_biases) {
    CUDA_1D_KERNEL_LOOP(i, p.elem_cnt) {
      const int32_t p_idx = i / (p.problem.m * p.problem.n);
      const int32_t y_idx = i % (p.problem.m * p.problem.n);
      const int32_t m_idx = y_idx / p.problem.n;
      const int32_t n_idx = y_idx % p.problem.n;
      p.buffer[p_idx].y[y_idx] = p.buffer[p_idx].b[n_idx];
    }
  }
  CUDA_1D_KERNEL_LOOP(i, p.n) {
    ptr_arr[i] = const_cast<T*>(p.buffer[i].x);
    ptr_arr[i + kMaxProblemBatch] = const_cast<T*>(p.buffer[i].w);
    ptr_arr[i + 2 * kMaxProblemBatch] = p.buffer[i].y;
  }
}

union CublasScalarParameter {
  double d;
  float s;
  half h;
};

CublasScalarParameter GetCublasScalarParameter(Scalar scalar, cudaDataType_t compute_type) {
  CublasScalarParameter sp{};
  if (compute_type == CUDA_R_64F) {
    sp.d = scalar.Value<double>();
  } else if (compute_type == CUDA_R_32F) {
    sp.s = scalar.Value<float>();
  } else if (compute_type == CUDA_R_16F) {
    sp.h = static_cast<half>(scalar.Value<float>());
  } else {
    UNIMPLEMENTED();
  }
  return sp;
}

template<typename T>
void ApplyGroup(const Problem& problem, std::vector<Buffer<T>> ptrs, bool has_biases,
                void* workspace, ep::Stream* stream) {
  Param<T> params(problem, ptrs);
  void** ptr_arr = reinterpret_cast<void**>(workspace);
  if (has_biases) {
    RUN_CUDA_KERNEL((InitPtrAndApplyBias<T, true>), stream, params.elem_cnt, params, ptr_arr);
  } else {
    RUN_CUDA_KERNEL((InitPtrAndApplyBias<T, false>), stream, params.n, params, ptr_arr);
  }
  float alpha = 1.0;
  float beta = has_biases ? 1.0 : 0.0;
  cudaDataType_t data_type{};
  cudaDataType_t compute_type{};
  if (std::is_same<T, half>::value) {
    data_type = CUDA_R_16F;
    const bool allow_half_accumulation =
        ParseBooleanFromEnv("ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION", false);
    if (allow_half_accumulation) {
      compute_type = CUDA_R_16F;
    } else {
      compute_type = CUDA_R_32F;
    }
  } else if (std::is_same<T, float>::value) {
    data_type = CUDA_R_32F;
    compute_type = CUDA_R_32F;
  } else {
    UNIMPLEMENTED();
  }
  auto sp_alpha = GetCublasScalarParameter(alpha, compute_type);
  auto sp_beta = GetCublasScalarParameter(beta, compute_type);
  OF_CUBLAS_CHECK(cublasGemmBatchedEx(
      stream->As<ep::CudaStream>()->cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, problem.n, problem.m,
      problem.k, &sp_alpha, ptr_arr + kMaxProblemBatch, data_type, problem.k, ptr_arr, data_type,
      problem.k, &sp_beta, ptr_arr + 2 * kMaxProblemBatch, data_type, problem.n, params.n,
      compute_type, CUBLAS_GEMM_DEFAULT));
}

template<typename T>
class GroupedMatmulBiasKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GroupedMatmulBiasKernel() = default;
  ~GroupedMatmulBiasKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    HashMap<Problem, std::vector<Buffer<T>>> groups;
    const int32_t input_size = ctx->input_size("xs");
    CHECK_EQ(ctx->input_size("weights"), input_size);
    const bool has_biases = ctx->has_input("biases", 0);
    if (has_biases) { CHECK_EQ(ctx->input_size("biases"), input_size); }
    CHECK_EQ(ctx->output_size("ys"), input_size);
    for (int32_t i = 0; i < input_size; ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("xs", i);
      const user_op::Tensor* w = ctx->Tensor4ArgNameAndIndex("weights", i);
      const user_op::Tensor* b = has_biases ? ctx->Tensor4ArgNameAndIndex("biases", i) : nullptr;
      user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("ys", i);
      CHECK_GE(x->shape_view().NumAxes(), 2);
      const int64_t k = x->shape_view().At(x->shape_view().NumAxes() - 1);
      const int64_t m = x->shape_view().elem_cnt() / k;
      CHECK_EQ(w->shape_view().NumAxes(), 2);
      CHECK_EQ(w->shape_view().At(1), k);
      const int64_t n = w->shape_view().At(0);
      if (has_biases) {
        CHECK_EQ(b->shape_view().NumAxes(), 1);
        CHECK_EQ(b->shape_view().At(0), n);
      }
      CHECK_EQ(y->shape_view().NumAxes(), x->shape_view().NumAxes());
      CHECK_EQ(y->shape_view().At(y->shape_view().NumAxes() - 1), n);
      for (int32_t j = 0; j < y->shape_view().NumAxes() - 1; ++j) {
        CHECK_EQ(y->shape_view().At(j), x->shape_view().At(j));
      }
      groups[Problem(m, n, k)].push_back(Buffer<T>{
          x->dptr<T>(), w->dptr<T>(), has_biases ? b->dptr<T>() : nullptr, y->mut_dptr<T>()});
    }
    void* workspace = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0)->mut_dptr();
    for (const auto& group : groups) {
      ApplyGroup<T>(group.first, group.second, has_biases, workspace, ctx->stream());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GROUPED_MATMUL_BIAS_KERNEL_GPU(cpp_type, data_type)    \
  REGISTER_USER_KERNEL("grouped_matmul_bias")                           \
      .SetCreateFn<GroupedMatmulBiasKernel<cpp_type>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)  \
                       && (user_op::HobDataType("ys", 0) == data_type)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {     \
        return kMaxProblemBatch * 3 * sizeof(void*);                    \
      });                                                               \
  ;

REGISTER_GROUPED_MATMUL_BIAS_KERNEL_GPU(float, DataType::kFloat)
REGISTER_GROUPED_MATMUL_BIAS_KERNEL_GPU(half, DataType::kFloat16)

}  // namespace

}  // namespace oneflow
