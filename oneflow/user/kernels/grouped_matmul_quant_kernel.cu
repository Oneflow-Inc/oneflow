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
#ifdef WITH_CUTLASS_EXTENSION

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/scalar.h"

#include <cutlass/library/library.h>
#include <cutlass/library/operation_table.h>
#include <cutlass/library/cutlass_extension_library.h>
#include "oneflow/user/kernels/cutlass_gemm_array_tuner.h"
#include <nlohmann/json.hpp>

namespace oneflow {

struct GemmProblem {
  GemmProblem(int64_t m, int64_t n, int64_t k) : m(m), n(n), k(k) {}
  int64_t m;
  int64_t n;
  int64_t k;
};

inline bool operator==(const GemmProblem& lhs, const GemmProblem& rhs) {
  return lhs.m == rhs.m && lhs.n == rhs.n && lhs.k == rhs.k;
}

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::GemmProblem> {
  std::size_t operator()(const oneflow::GemmProblem& p) const {
    return oneflow::Hash<int64_t, int64_t, int64_t>(p.m, p.n, p.k);
  }
};

}  // namespace std

namespace oneflow {

namespace {

constexpr int64_t kMaxProblemBatch = 32;

template<typename T>
struct Buffer {
  const int8_t* a;
  const int8_t* b;
  const int8_t* in_zero_point;
  const float* in_scale;
  const T* weight_scale;
  const T* weight_acc;
  const T* scale;
  const T* biase;
  const T* _add_to_output;
  T* output;
};

template<typename T>
struct Param {
  Param(const GemmProblem& problem, std::vector<Buffer<T>> buffers)
      : problem(problem), batch_count(buffers.size()) {
    std::copy(buffers.cbegin(), buffers.cend(), buffer);
  }
  GemmProblem problem;
  Buffer<T> buffer[kMaxProblemBatch];
  int batch_count;
};

template<typename T>
__global__ void InitPtr(Param<T> p, void** ptr_arr) {
  CUDA_1D_KERNEL_LOOP(i, p.batch_count) {
    ptr_arr[i] = const_cast<int8_t*>(p.buffer[i].a);
    ptr_arr[i + kMaxProblemBatch] = const_cast<int8_t*>(p.buffer[i].b);
    ptr_arr[i + 2 * kMaxProblemBatch] = const_cast<int8_t*>(p.buffer[i].in_zero_point);
    ptr_arr[i + 3 * kMaxProblemBatch] = const_cast<float*>(p.buffer[i].in_scale);
    ptr_arr[i + 4 * kMaxProblemBatch] = const_cast<T*>(p.buffer[i].weight_scale);
    ptr_arr[i + 5 * kMaxProblemBatch] = const_cast<T*>(p.buffer[i].weight_acc);
    ptr_arr[i + 6 * kMaxProblemBatch] = const_cast<T*>(p.buffer[i].scale);
    ptr_arr[i + 7 * kMaxProblemBatch] = const_cast<T*>(p.buffer[i].biase);
    ptr_arr[i + 8 * kMaxProblemBatch] = const_cast<T*>(p.buffer[i]._add_to_output);
    ptr_arr[i + 9 * kMaxProblemBatch] = p.buffer[i].output;
  }
}

template<typename T>
void ApplyGroup(user_op::KernelComputeContext* ctx, const cutlass::library::GemmFunctionalKey& key,
                const GemmProblem& problem, bool has_in_zero_points, bool has_sacles,
                bool has_biases, bool has_add_to_outputs, std::vector<Buffer<T>> ptrs,
                user_op::Tensor* tmp_buffer, ep::Stream* stream) {
  void* tmp_ptr = tmp_buffer->mut_dptr();
  void** ptr_arr = reinterpret_cast<void**>(tmp_ptr);
  void* workspace = tmp_buffer + kMaxProblemBatch * 10 * sizeof(void*);
  size_t workspace_size =
      tmp_buffer->shape_view().elem_cnt() - kMaxProblemBatch * 10 * sizeof(void*);
  Param<T> params(problem, ptrs);
  RUN_CUDA_KERNEL((InitPtr<T>), stream, params.batch_count, params, ptr_arr);

  cutlass::gemm::GemmCoord problem_size(problem.m, problem.n, problem.k);

  cutlass::library::GemmArrayScaleBiasFusionConfiguration configuraion;
  configuraion.problem_size = problem_size;
  configuraion.lda = problem_size.k();
  configuraion.ldb = problem_size.k();
  configuraion.ld_filter_scale = 0;
  configuraion.ld_filter_acc = 0;
  configuraion.ld_scale = 0;
  configuraion.ld_bias = 0;
  configuraion.ldr = problem_size.n();
  configuraion.ldd = problem_size.n();
  configuraion.batch_count = params.batch_count;

  cutlass::library::GemmArrayScaleBiasFusionArguments arguments;
  arguments.A = ptr_arr;
  arguments.B = ptr_arr + kMaxProblemBatch;
  arguments.D = ptr_arr + 9 * kMaxProblemBatch;
  arguments.P = nullptr;
  arguments.InScale = nullptr;
  arguments.FilterScale = nullptr;
  arguments.FilterAcc = nullptr;
  arguments.Scale = nullptr;
  arguments.Bias = nullptr;
  arguments.Residual = nullptr;
  if (has_in_zero_points) {
    arguments.P = ptr_arr + 2 * kMaxProblemBatch;
    arguments.InScale = ptr_arr + 3 * kMaxProblemBatch;
    arguments.FilterScale = ptr_arr + 4 * kMaxProblemBatch;
    arguments.FilterAcc = ptr_arr + 5 * kMaxProblemBatch;
  }
  if (has_sacles) { arguments.Scale = ptr_arr + 6 * kMaxProblemBatch; }
  if (has_biases) { arguments.Bias = ptr_arr + 7 * kMaxProblemBatch; }
  if (has_add_to_outputs) { arguments.Residual = ptr_arr + 8 * kMaxProblemBatch; }

  auto* cuda_stream = stream->As<ep::CudaStream>();
  const cutlass::library::Operation* operation = nullptr;

  operation = [&]() -> const cutlass::library::Operation* {
    const std::string& tuning_cache = ctx->Attr<std::string>("tuning_cache");
    if (tuning_cache.empty()) { return nullptr; }
    auto tuning_cache_object = nlohmann::json::parse(tuning_cache);
    if (!tuning_cache_object.is_object()) { return nullptr; }
    auto it = tuning_cache_object.find("cutlass");
    if (it == tuning_cache_object.end()) { return nullptr; }
    if (!it->is_string()) { return nullptr; }
    const std::string name = *it;
    return CutlassGemmArrayTuner().GetOperation(name, cuda_stream, key, configuraion, arguments,
                                                workspace, workspace_size);
  }();
  if (!operation) {
    operation = CutlassGemmArrayTuner().FindOperation(cuda_stream, key, configuraion, arguments,
                                                      workspace, workspace_size);
  }
  CHECK(operation != nullptr);
  const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
  std::vector<uint8_t> host_workspace(host_workspace_size, 0);

  auto init_status = operation->initialize(&configuraion, host_workspace.data(), workspace,
                                           cuda_stream->cuda_stream());
  CHECK(init_status == cutlass::Status::kSuccess);
  auto run_status =
      operation->run(&arguments, host_workspace.data(), workspace, cuda_stream->cuda_stream());
  CHECK(run_status == cutlass::Status::kSuccess);
}

template<typename OutType>
class GroupedMatmulQuantKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GroupedMatmulQuantKernel() = default;
  ~GroupedMatmulQuantKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    HashMap<GemmProblem, std::vector<Buffer<OutType>>> groups;
    const int32_t input_size = ctx->input_size("as");
    CHECK_EQ(ctx->input_size("bs"), input_size);
    const bool has_in_zero_points = ctx->has_input("in_zero_points", 0);
    const bool has_sacles = ctx->has_input("scales", 0);
    const bool has_biases = ctx->has_input("biases", 0);
    const bool has_add_to_outputs = ctx->has_input("_add_to_outputs", 0);

    cutlass::library::GemmFunctionalKey key(
        cutlass::library::Provider::kCUTLASS, cutlass::library::GemmKind::kGemm,
        cutlass::library::NumericTypeID::kS32,         // element_compute
        cutlass::library::NumericTypeID::kS32,         // element_scalar
        cutlass::library::NumericTypeID::kS8,          // element_A
        cutlass::library::LayoutTypeID::kRowMajor,     // layout_A
        cutlass::library::ComplexTransform::kNone,     // transform_A
        cutlass::library::NumericTypeID::kS8,          // element_B
        cutlass::library::LayoutTypeID::kColumnMajor,  // layout_B
        cutlass::library::ComplexTransform::kNone,     // transform_B
        cutlass::library::NumericTypeID::kS32,         // element_C
        cutlass::library::LayoutTypeID::kRowMajor,     // layout_C
        cutlass::library::NumericTypeID::kS32,         // element_D
        cutlass::library::LayoutTypeID::kRowMajor      // layout_D
    );

    if (GetDataType<OutType>::value == DataType::kFloat) {
      key.element_scalar = cutlass::library::NumericTypeID::kF32;
      key.element_C = cutlass::library::NumericTypeID::kF32;
      key.element_D = cutlass::library::NumericTypeID::kF32;
    } else if (GetDataType<OutType>::value == DataType::kFloat16) {
      key.element_scalar = cutlass::library::NumericTypeID::kF32;
      key.element_C = cutlass::library::NumericTypeID::kF16;
      key.element_D = cutlass::library::NumericTypeID::kF16;
    }

    for (int32_t i = 0; i < input_size; ++i) {
      const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("as", i);
      const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("bs", i);
      const user_op::Tensor* in_zero_point = ctx->Tensor4ArgNameAndIndex("in_zero_points", i);
      const user_op::Tensor* in_scale = ctx->Tensor4ArgNameAndIndex("in_scales", i);
      const user_op::Tensor* weight_scale = ctx->Tensor4ArgNameAndIndex("weight_scales", i);
      const user_op::Tensor* weight_acc = ctx->Tensor4ArgNameAndIndex("weight_accs", i);
      const user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scales", i);
      const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("biases", i);
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_outputs", i);
      user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("outputs", i);

      CHECK_GE(a->shape_view().NumAxes(), 2);
      const int64_t k = a->shape_view().At(a->shape_view().NumAxes() - 1);
      const int64_t m = a->shape_view().elem_cnt() / k;
      const int64_t n = b->shape_view().At(0);

      CHECK_EQ(output->shape_view().NumAxes(), a->shape_view().NumAxes());
      CHECK_EQ(output->shape_view().At(output->shape_view().NumAxes() - 1), n);
      for (int32_t j = 0; j < output->shape_view().NumAxes() - 1; ++j) {
        CHECK_EQ(output->shape_view().At(j), a->shape_view().At(j));
      }
      const int8_t* a_ptr = a->dptr<int8_t>();
      const int8_t* b_ptr = b->dptr<int8_t>();
      const int8_t* in_zero_point_ptr =
          has_in_zero_points ? in_zero_point->dptr<int8_t>() : nullptr;
      const float* in_scale_ptr = has_in_zero_points ? in_scale->dptr<float>() : nullptr;
      const OutType* weight_scale_ptr =
          has_in_zero_points ? weight_scale->dptr<OutType>() : nullptr;
      const OutType* weight_acc_ptr = has_in_zero_points ? weight_acc->dptr<OutType>() : nullptr;
      const OutType* scale_ptr = has_sacles ? scale->dptr<OutType>() : nullptr;
      const OutType* bias_ptr = has_biases ? bias->dptr<OutType>() : nullptr;
      const OutType* add_to_output_ptr =
          has_add_to_outputs ? add_to_output->dptr<OutType>() : nullptr;
      OutType* output_ptr = output->mut_dptr<OutType>();

      groups[GemmProblem(m, n, k)].push_back(
          Buffer<OutType>{a_ptr, b_ptr, in_zero_point_ptr, in_scale_ptr, weight_scale_ptr,
                          weight_acc_ptr, scale_ptr, bias_ptr, add_to_output_ptr, output_ptr});
    }
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    for (const auto& group : groups) {
      for (size_t i = 0; i < group.second.size(); i += kMaxProblemBatch) {
        std::vector<Buffer<OutType>> ptrs(
            {group.second.begin() + i,
             group.second.begin() + i
                 + std::min<size_t>(group.second.size() - i, kMaxProblemBatch)});
        ApplyGroup<OutType>(ctx, key, group.first, has_in_zero_points, has_sacles, has_biases,
                            has_add_to_outputs, ptrs, tmp_buffer, ctx->stream());
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GROUPED_MATMUL_BIAS_KERNEL_GPU(out_cpp_type, out_data_type)     \
  REGISTER_USER_KERNEL("grouped_matmul_quant")                                   \
      .SetCreateFn<GroupedMatmulQuantKernel<out_cpp_type>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)           \
                       && (user_op::HobDataType("as", 0) == DataType::kInt8)     \
                       && (user_op::HobDataType("bs", 0) == DataType::kInt8)     \
                       && (user_op::HobDataType("outputs", 0) == out_data_type)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {              \
        return kMaxProblemBatch * 10 * sizeof(void*) + 3 * 1024 * 1024;          \
      });

REGISTER_GROUPED_MATMUL_BIAS_KERNEL_GPU(half, DataType::kFloat16)
REGISTER_GROUPED_MATMUL_BIAS_KERNEL_GPU(float, DataType::kFloat)

}  // namespace

}  // namespace oneflow

#endif  // WITH_CUTLASS_EXTENSION
