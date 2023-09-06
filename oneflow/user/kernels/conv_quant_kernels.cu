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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/user/kernels/cutlass_conv_tuner.h"

#include <cutlass/library/library.h>
#include <cutlass/library/cutlass_extension_library.h>
#include <nlohmann/json.hpp>

namespace oneflow {

namespace {

template<typename Configuration, typename Arguments>
void LaunchConvQuantOpImpl(user_op::KernelComputeContext* ctx,
                           const cutlass::library::ConvFunctionalKey& key,
                           const Configuration& configuraion, const Arguments& arguments) {
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
    return CutlassConvTuner().GetConv2dOperation(name, stream, key, configuraion, arguments,
                                                 tmp_buffer->mut_dptr(),
                                                 tmp_buffer->shape_view().elem_cnt());
  }();
  if (!operation) {
    operation = CutlassConvTuner().FindConv2dOperation(stream, key, configuraion, arguments,
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

void LaunchConv2dQuantOp(user_op::KernelComputeContext* ctx,
                         const cutlass::library::ConvFunctionalKey& key,
                         const cutlass::conv::Conv2dProblemSize& problem_size,
                         const user_op::Tensor* in, const user_op::Tensor* weight,
                         const user_op::Tensor* in_zero_point, const user_op::Tensor* in_scale,
                         const user_op::Tensor* weight_scale, const user_op::Tensor* weight_acc,
                         const user_op::Tensor* scale, const user_op::Tensor* bias,
                         const user_op::Tensor* add_to_output, user_op::Tensor* out) {
  cutlass::library::Conv2dScaleBiasFusionConfiguration configuraion;
  configuraion.split_k_mode = cutlass::conv::SplitKMode::kSerial;
  configuraion.problem_size = problem_size;
  configuraion.stride_a = {problem_size.C, problem_size.W * problem_size.C,
                           problem_size.H * problem_size.W * problem_size.C};
  configuraion.stride_b = {problem_size.C, problem_size.S * problem_size.C,
                           problem_size.R * problem_size.S * problem_size.C};
  if (add_to_output) {
    configuraion.stride_residual = {problem_size.K, problem_size.Q * problem_size.K,
                                    problem_size.P * problem_size.Q * problem_size.K};
  }
  cutlass::library::ConvScaleBiasFusionArguments arguments;
  arguments.A = in->dptr();
  arguments.B = weight->dptr();
  arguments.P = in_zero_point->dptr();
  arguments.D = out->mut_dptr();

  arguments.InScale = nullptr;
  arguments.FilterScale = nullptr;
  arguments.FilterAcc = nullptr;
  arguments.Scale = nullptr;
  arguments.Bias = nullptr;
  arguments.Residual = nullptr;

  if (in_scale) { arguments.InScale = in_scale->dptr(); }
  if (weight_scale) { arguments.FilterScale = weight_scale->dptr(); }
  if (weight_acc) { arguments.FilterAcc = weight_acc->dptr(); }
  if (scale) { arguments.Scale = scale->dptr(); }
  if (bias) { arguments.Bias = bias->dptr(); }
  if (add_to_output) { arguments.Residual = add_to_output->dptr(); }

  LaunchConvQuantOpImpl(ctx, key, configuraion, arguments);
}

class Conv2dQuantKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  Conv2dQuantKernel() = default;
  ~Conv2dQuantKernel() override = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");

    const int n = in->shape_view().At(0);
    const int h = in->shape_view().At(1);
    const int w = in->shape_view().At(2);
    const int c = in->shape_view().At(3);

    const int k = weight->shape_view().At(0);
    const int r = weight->shape_view().At(1);
    const int s = weight->shape_view().At(2);
    CHECK_EQ(weight->shape_view().At(3), c);

    const int p = out->shape_view().At(1);
    const int q = out->shape_view().At(2);

    cutlass::library::ConvFunctionalKey key(
        cutlass::library::Provider::kCUTLASS, cutlass::library::ConvKind::kFprop,
        cutlass::library::NumericTypeID::kS8, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kS8, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kS32, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kS32, cutlass::library::NumericTypeID::kS32);
    if (out->data_type() == DataType::kFloat) {
      key.element_C = cutlass::library::NumericTypeID::kF32;
      key.element_compute = cutlass::library::NumericTypeID::kF32;
    } else if (out->data_type() == DataType::kFloat16) {
      key.element_C = cutlass::library::NumericTypeID::kF16;
      key.element_compute = cutlass::library::NumericTypeID::kF32;
    }
    cutlass::conv::Conv2dProblemSize problem_size(
        n, h, w, c, k, r, s, p, q, padding_before.at(0), padding_before.at(1), strides.at(0),
        strides.at(1), dilation_rate.at(0), dilation_rate.at(1),
        cutlass::conv::Mode::kCrossCorrelation);

    const user_op::Tensor* in_zero_point = ctx->Tensor4ArgNameAndIndex("in_zero_point", 0);
    const user_op::Tensor* in_scale = ctx->Tensor4ArgNameAndIndex("in_scale", 0);
    const user_op::Tensor* weight_scale = ctx->Tensor4ArgNameAndIndex("weight_scale", 0);
    const user_op::Tensor* weight_acc = ctx->Tensor4ArgNameAndIndex("weight_acc", 0);
    const user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);

    LaunchConv2dQuantOp(ctx, key, problem_size, in, weight, in_zero_point, in_scale, weight_scale,
                        weight_acc, scale, bias, add_to_output, out);
  }
};

REGISTER_USER_KERNEL("conv2d_quant")
    .SetCreateFn<Conv2dQuantKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobAttr<std::string>("data_format") == "channels_last")
                     && (user_op::HobAttr<int32_t>("groups") == 1)
                     && (user_op::HobDataType("in", 0) == DataType::kInt8))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      // use static workspace size
      return 128 * 1024 * 1024;
    })
    .SetPriority(user_op::kKernelPriorityOptimized);

}  // namespace

}  // namespace oneflow

#endif  // WITH_CUTLASS_EXTENSION
