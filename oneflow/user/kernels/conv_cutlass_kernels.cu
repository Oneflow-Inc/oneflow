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

#ifdef WITH_CUTLASS

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/user/kernels/cutlass_conv_tuner.h"
#include <cutlass/library/handle.h>
#include <cutlass/library/library.h>
#include <cutlass/library/singleton.h>
#include <nlohmann/json.hpp>

namespace oneflow {

namespace {

class Conv2dCutlassKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  Conv2dCutlassKernel() = default;
  ~Conv2dCutlassKernel() override = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
    CHECK(add_to_output == nullptr);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

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

    auto* stream = ctx->stream()->As<ep::CudaStream>();

    cutlass::library::ConvFunctionalKey key(
        cutlass::library::Provider::kCUTLASS, cutlass::library::ConvKind::kFprop,
        cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kF32, cutlass::library::NumericTypeID::kF32);

    const bool allow_half_accumulation =
        ParseBooleanFromEnv("ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION", false);

    if (allow_half_accumulation) {
      key.element_accumulator = cutlass::library::NumericTypeID::kF16;
      key.element_compute = cutlass::library::NumericTypeID::kF16;
    }

    cutlass::conv::Conv2dProblemSize problem_size(
        n, h, w, c, k, r, s, p, q, padding_before.at(0), padding_before.at(1), strides.at(0),
        strides.at(1), dilation_rate.at(0), dilation_rate.at(1),
        cutlass::conv::Mode::kCrossCorrelation);
    cutlass::library::Conv2dConfiguration configuraion;
    configuraion.split_k_mode = cutlass::conv::SplitKMode::kSerial;
    configuraion.problem_size = problem_size;
    configuraion.stride_a = {c, w * c, h * w * c};
    configuraion.stride_b = {c, s * c, r * s * c};
    configuraion.stride_c = {0, 0, 0};

    cutlass::library::ConvArguments arguments;
    arguments.A = in->dptr();
    arguments.B = weight->dptr();
    arguments.reordered_B = nullptr;
    if (bias == nullptr) {
      arguments.C = nullptr;
    } else {
      arguments.C = bias->dptr();
    }
    arguments.D = out->mut_dptr();

    union SP {
      float f;
      half h;
    };

    SP alpha;
    SP beta;

    if (allow_half_accumulation) {
      alpha.h = static_cast<half>(1.0F);
      if (bias == nullptr) {
        beta.h = static_cast<half>(0.0F);
      } else {
        beta.h = static_cast<half>(1.0F);
      }
    } else {
      alpha.f = 1.0F;
      if (bias == nullptr) {
        beta.f = 0.0F;
      } else {
        beta.f = 1.0F;
      }
    }
    arguments.alpha = &alpha;
    arguments.beta = &beta;
    arguments.pointer_mode = cutlass::library::ScalarPointerMode::kHost;
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
      return CutlassConvTuner::Get().GetConv2dOperation(name, stream, key, configuraion, arguments,
                                                        tmp_buffer->mut_dptr(),
                                                        tmp_buffer->shape_view().elem_cnt());
    }();
    if (!operation) {
      operation = CutlassConvTuner::Get().FindConv2dOperation(stream, key, configuraion, arguments,
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
};

REGISTER_USER_KERNEL("conv2d")
    .SetCreateFn<Conv2dCutlassKernel>()
    .SetIsMatchedHob(
        (user_op::HobDeviceType() == DeviceType::kCUDA)
        && (user_op::HobAttr<std::string>("data_format") == "channels_last")
        && (user_op::HobAttr<int32_t>("groups") == 1)
        && (user_op::HobDataType("in", 0) == DataType::kFloat16)
        // Compatible with typo `KERENL`
        && ((user_op::HobEnvBool("ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL", false) == true)
            || (user_op::HobEnvBool("ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL", false) == true)))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      // use static workspace size
      return 128 * 1024 * 1024;
    })
    .SetPriority(user_op::kKernelPriorityOptimized);

}  // namespace

}  // namespace oneflow

#endif  // WITH_CUTLASS
