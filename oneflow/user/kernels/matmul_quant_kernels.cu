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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

#include "oneflow/user/kernels/cutlass_gemm_tuner.h"

#include <cutlass/library/library.h>
#include <cutlass/library/operation_table.h>
#include <cutlass/library/cutlass_extension_library.h>
#include <nlohmann/json.hpp>

namespace oneflow {

namespace {

void LaunchMatmulQuantOp(user_op::KernelComputeContext* ctx,
                         const cutlass::library::GemmFunctionalKey& key,
                         const cutlass::gemm::GemmCoord& problem_size, const user_op::Tensor* a,
                         const user_op::Tensor* b, const user_op::Tensor* in_zero_point,
                         const user_op::Tensor* in_scale, const user_op::Tensor* weight_scale,
                         const user_op::Tensor* weight_acc, const user_op::Tensor* scale,
                         const user_op::Tensor* bias, const user_op::Tensor* add_to_output,
                         user_op::Tensor* out) {
  cutlass::library::GemmScaleBiasFusionConfiguration configuraion;
  configuraion.problem_size = problem_size;
  configuraion.lda = problem_size.k();
  configuraion.ldb = problem_size.k();
  configuraion.ld_filter_scale = 0;
  configuraion.ld_filter_acc = 0;
  configuraion.ld_scale = 0;
  configuraion.ld_bias = 0;
  configuraion.ldr = problem_size.n();
  configuraion.ldd = problem_size.n();
  configuraion.split_k_slices = 1;
  // if (problem_size.m() <= 2 && problem_size.k() >= 4096) { configuraion.split_k_slices = 16; }

  cutlass::library::GemmScaleBiasFusionArguments arguments;
  arguments.A = a->dptr();
  arguments.B = b->dptr();
  arguments.D = out->mut_dptr();
  arguments.P = nullptr;
  arguments.InScale = nullptr;
  arguments.FilterScale = nullptr;
  arguments.FilterAcc = nullptr;
  arguments.Scale = nullptr;
  arguments.Bias = nullptr;
  arguments.Residual = nullptr;

  if (in_zero_point) { arguments.P = in_zero_point->dptr(); }
  if (in_scale) { arguments.InScale = in_scale->dptr(); }
  if (weight_scale) { arguments.FilterScale = weight_scale->dptr(); }
  if (weight_acc) { arguments.FilterAcc = weight_acc->dptr(); }
  if (scale) { arguments.Scale = scale->dptr(); }
  if (bias) { arguments.Bias = bias->dptr(); }
  if (add_to_output) { arguments.Residual = add_to_output->dptr(); }

  user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
  auto* stream = ctx->stream()->As<ep::CudaStream>();
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
    return CutlassGemmTuner().GetOperation(name, stream, key, configuraion, arguments,
                                           tmp_buffer->mut_dptr(),
                                           tmp_buffer->shape_view().elem_cnt());
  }();
  if (!operation) {
    operation = CutlassGemmTuner().FindOperation(stream, key, configuraion, arguments,
                                                 tmp_buffer->mut_dptr(),
                                                 tmp_buffer->shape_view().elem_cnt());
  }
  CHECK(operation != nullptr);
  const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
  std::vector<uint8_t> host_workspace(host_workspace_size, 0);
  auto init_status = operation->initialize(&configuraion, host_workspace.data(),
                                           tmp_buffer->mut_dptr(), stream->cuda_stream());
  CHECK(init_status == cutlass::Status::kSuccess);
  auto run_status = operation->run(&arguments, host_workspace.data(), tmp_buffer->mut_dptr(),
                                   stream->cuda_stream());
  CHECK(run_status == cutlass::Status::kSuccess);
}

}  // namespace

class MatmulQuantKernel final : public user_op::OpKernel {
 public:
  MatmulQuantKernel() = default;
  ~MatmulQuantKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    CHECK(!ctx->Attr<bool>("transpose_a"));
    CHECK(ctx->Attr<bool>("transpose_b"));

    int64_t dim_a = a->shape_view().NumAxes();
    const int m = a->shape_view().Count(0, dim_a - 1);
    const int k = a->shape_view().At(dim_a - 1);
    const int n = b->shape_view().At(0);

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

    if (a->data_type() == DataType::kFloat16) {
      key.element_A = cutlass::library::NumericTypeID::kF16;
    }

    if (out->data_type() == DataType::kFloat) {
      key.element_scalar = cutlass::library::NumericTypeID::kF32;
      key.element_C = cutlass::library::NumericTypeID::kF32;
      key.element_D = cutlass::library::NumericTypeID::kF32;
    } else if (out->data_type() == DataType::kFloat16) {
      key.element_scalar = cutlass::library::NumericTypeID::kF32;
      key.element_C = cutlass::library::NumericTypeID::kF16;
      key.element_D = cutlass::library::NumericTypeID::kF16;
    }

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    const user_op::Tensor* in_zero_point = ctx->Tensor4ArgNameAndIndex("in_zero_point", 0);
    const user_op::Tensor* in_scale = ctx->Tensor4ArgNameAndIndex("in_scale", 0);
    const user_op::Tensor* weight_scale = ctx->Tensor4ArgNameAndIndex("weight_scale", 0);
    const user_op::Tensor* weight_acc = ctx->Tensor4ArgNameAndIndex("weight_acc", 0);
    const user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);

    LaunchMatmulQuantOp(ctx, key, problem_size, a, b, in_zero_point, in_scale, weight_scale,
                        weight_acc, scale, bias, add_to_output, out);
  }
};

#define REGISTER_MATMUL_QUANT_KERNEL(data_type)                                                  \
  REGISTER_USER_KERNEL("matmul_quant")                                                           \
      .SetCreateFn<MatmulQuantKernel>()                                                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                           \
                       && (user_op::HobDataType("a", 0) == data_type)                            \
                       && (user_op::HobDataType("b", 0) == DataType::kInt8))                     \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t { return 128 * 1024 * 1024; }) \
      .SetPriority(user_op::kKernelPriorityOptimized);

REGISTER_MATMUL_QUANT_KERNEL(DataType::kInt8)
REGISTER_MATMUL_QUANT_KERNEL(DataType::kFloat16)

}  // namespace oneflow

#endif  // WITH_CUTLASS_EXTENSION
