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

#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/user/kernels/cutlass_conv_tuner.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

namespace {

constexpr size_t kMaxWorkspaceSize = 128 * 1024 * 1024;  // 128MB

class CutlassConvTuningWarmupPass final : public JobPass {
 public:
  CutlassConvTuningWarmupPass() = default;
  ~CutlassConvTuningWarmupPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> CutlassConvTuningWarmupPass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!ParseBooleanFromEnv("ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL", false)) {
    return Maybe<void>::Ok();
  }
  if (!ParseBooleanFromEnv("ONEFLOW_KERENL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP", false)) {
    return Maybe<void>::Ok();
  }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](const OpNode* node) -> Maybe<void> {
    const OperatorConf& op_conf = node->op().op_conf();
    if (!op_conf.has_user_conf()) { return Maybe<void>::Ok(); }
    if (op_conf.user_conf().op_type_name() != "conv2d") { return Maybe<void>::Ok(); }
    if (node->parallel_desc().device_type() != DeviceType::kCUDA) { return Maybe<void>::Ok(); }
    if (node->parallel_desc().parallel_num() != 1) { return Maybe<void>::Ok(); }
    if (!node->parallel_desc().containing_current_rank()) { return Maybe<void>::Ok(); }
    user_op::UserOpConfWrapper conv2d_op(op_conf);
    if (conv2d_op.attr<std::string>("data_format") != "channels_last") { return Maybe<void>::Ok(); }
    if (conv2d_op.attr<int32_t>("groups") != 1) { return Maybe<void>::Ok(); }
    VLOG(3) << "Tuning " << op_conf.name();
    const auto& in_desc = node->LogicalBlobDesc4Lbi(GenLogicalBlobId(conv2d_op.input("in", 0)));
    if (in_desc.data_type() != DataType::kFloat16) { return Maybe<void>::Ok(); }
    const auto& weight_desc =
        node->LogicalBlobDesc4Lbi(GenLogicalBlobId(conv2d_op.input("weight", 0)));
    const auto& out_desc = node->LogicalBlobDesc4Lbi(GenLogicalBlobId(conv2d_op.output("out", 0)));

    const auto& padding_before = conv2d_op.attr<std::vector<int32_t>>("padding_before");
    const auto& dilation_rate = conv2d_op.attr<std::vector<int32_t>>("dilation_rate");
    const auto& strides = conv2d_op.attr<std::vector<int32_t>>("strides");

    const int n = in_desc.shape().At(0);
    const int h = in_desc.shape().At(1);
    const int w = in_desc.shape().At(2);
    const int c = in_desc.shape().At(3);

    const int k = weight_desc.shape().At(0);
    const int r = weight_desc.shape().At(1);
    const int s = weight_desc.shape().At(2);
    CHECK_EQ(weight_desc.shape().At(3), c);

    const int p = out_desc.shape().At(1);
    const int q = out_desc.shape().At(2);

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

    auto device = Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCUDA, 0);
    ep::Stream* stream = device->CreateStream();

    void* workspace = nullptr;
    OF_CUDA_CHECK(cudaMalloc(&workspace, kMaxWorkspaceSize));
    void* x_ptr = nullptr;
    void* w_ptr = nullptr;
    void* y_ptr = nullptr;
    void* bias_ptr = nullptr;
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
    OF_CUDA_CHECK(cudaMalloc(&x_ptr, in_desc.ByteSizeOfBlobBody()));
    arguments.A = x_ptr;
    OF_CUDA_CHECK(cudaMalloc(&w_ptr, weight_desc.ByteSizeOfBlobBody()));
    arguments.B = w_ptr;
    arguments.reordered_B = nullptr;
    if (conv2d_op.has_input("bias", 0)) {
      const auto& bias_desc =
          node->LogicalBlobDesc4Lbi(GenLogicalBlobId(conv2d_op.input("bias", 0)));
      OF_CUDA_CHECK(cudaMalloc(&bias_ptr, bias_desc.ByteSizeOfBlobBody()));
      arguments.C = bias_ptr;
    } else {
      arguments.C = nullptr;
    }
    OF_CUDA_CHECK(cudaMalloc(&y_ptr, out_desc.ByteSizeOfBlobBody()));
    arguments.D = y_ptr;
    union SP {
      float f{};
      half h;
    };

    SP alpha;
    SP beta;

    if (allow_half_accumulation) {
      alpha.h = static_cast<half>(1.0F);
      if (bias_ptr == nullptr) {
        beta.h = static_cast<half>(0.0F);
      } else {
        beta.h = static_cast<half>(1.0F);
      }
    } else {
      alpha.f = 1.0F;
      if (bias_ptr == nullptr) {
        beta.f = 0.0F;
      } else {
        beta.f = 1.0F;
      }
    }
    arguments.alpha = &alpha;
    arguments.beta = &beta;
    arguments.pointer_mode = cutlass::library::ScalarPointerMode::kHost;

    const cutlass::library::Operation* operation = CutlassConvTuner::Get().FindConv2dOperation(
        stream->As<ep::CudaStream>(), key, configuraion, arguments, workspace, kMaxWorkspaceSize);
    if (operation != nullptr) { VLOG(3) << "Fastest operation: " << operation->description().name; }

    OF_CUDA_CHECK(cudaFree(workspace));
    OF_CUDA_CHECK(cudaFree(x_ptr));
    OF_CUDA_CHECK(cudaFree(w_ptr));
    OF_CUDA_CHECK(cudaFree(y_ptr));
    OF_CUDA_CHECK(cudaFree(bias_ptr));
    device->DestroyStream(stream);

    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("CutlassConvTuningWarmupPass", CutlassConvTuningWarmupPass);

}  // namespace oneflow

#endif  // WITH_CUTLASS
