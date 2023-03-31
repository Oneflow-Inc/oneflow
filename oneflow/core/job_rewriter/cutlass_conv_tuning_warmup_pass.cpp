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
#include <nlohmann/json.hpp>

namespace oneflow {

namespace {

constexpr size_t kMaxWorkspaceSize = 128 * 1024 * 1024;   // 128MB
constexpr size_t kBufferMallocAlign = 128 * 1024 * 1024;  // 128MB

class CutlassConvTuningWarmupPass final : public JobPass {
 public:
  CutlassConvTuningWarmupPass() = default;
  ~CutlassConvTuningWarmupPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> CutlassConvTuningWarmupPass::Apply(Job* job, JobPassCtx* ctx) const {
  // Compatible with typo `KERENL`
  if (!ParseBooleanFromEnv("ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL",
                           ParseBooleanFromEnv("ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL", false))) {
    return Maybe<void>::Ok();
  }
  if (!ParseBooleanFromEnv(
          "ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP",
          ParseBooleanFromEnv("ONEFLOW_KERENL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP", false))) {
    return Maybe<void>::Ok();
  }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);

  auto device = Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCUDA, 0);
  ep::Stream* stream = device->CreateStream();
  void* workspace = nullptr;
  char* buffer = nullptr;
  size_t buffer_size = 0;
  OF_CUDA_CHECK(cudaMalloc(&workspace, kMaxWorkspaceSize));
  std::vector<OperatorConf> op_confs;
  op_graph.ForEachNode([&](const OpNode* node) {
    const OperatorConf& op_conf = node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    if (op_conf.user_conf().op_type_name() != "conv2d") { return; }
    if (node->parallel_desc().device_type() != DeviceType::kCUDA) { return; }
    if (node->parallel_desc().parallel_num() != 1) { return; }
    if (!node->parallel_desc().containing_current_rank()) { return; }
    user_op::UserOpConfWrapper conv2d_op(op_conf);
    if (conv2d_op.attr<std::string>("data_format") != "channels_last") { return; }
    if (conv2d_op.attr<int32_t>("groups") != 1) { return; }
    VLOG(3) << "Tuning " << op_conf.name();
    const auto& in_desc = node->LogicalBlobDesc4Lbi(GenLogicalBlobId(conv2d_op.input("in", 0)));
    if (in_desc.data_type() != DataType::kFloat16) { return; }
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

    const size_t x_size = GetCudaAlignedSize(in_desc.ByteSizeOfBlobBody());
    const size_t w_size = GetCudaAlignedSize(weight_desc.ByteSizeOfBlobBody());
    const size_t y_size = GetCudaAlignedSize(out_desc.ByteSizeOfBlobBody());
    size_t bias_size = 0;
    if (conv2d_op.has_input("bias", 0)) {
      bias_size =
          GetCudaAlignedSize(node->LogicalBlobDesc4Lbi(GenLogicalBlobId(conv2d_op.input("bias", 0)))
                                 .ByteSizeOfBlobBody());
    }
    const size_t total_buf_size = x_size + w_size + y_size + bias_size;
    if (total_buf_size > buffer_size) {
      size_t malloc_size = RoundUp(total_buf_size, kBufferMallocAlign);
      OF_CUDA_CHECK(cudaFree(buffer));
      OF_CUDA_CHECK(cudaMalloc(&buffer, malloc_size));
      buffer_size = malloc_size;
    }
    void* x_ptr = buffer;
    void* w_ptr = buffer + x_size;
    void* y_ptr = buffer + x_size + w_size;
    void* bias_ptr = nullptr;
    if (bias_size != 0) { bias_ptr = buffer + x_size + w_size + y_size; }

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
    arguments.A = x_ptr;
    arguments.B = w_ptr;
    arguments.reordered_B = nullptr;
    arguments.C = bias_ptr;
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
    if (operation != nullptr) {
      VLOG(3) << "Fastest operation: " << operation->description().name;
      nlohmann::json tuning_cache;
      tuning_cache["cutlass"] = operation->description().name;
      OperatorConf new_op_conf = op_conf;
      (*(*new_op_conf.mutable_user_conf()->mutable_attr())["tuning_cache"].mutable_at_string()) =
          tuning_cache.dump();
      op_confs.push_back(new_op_conf);
    }
  });
  job_builder.MutOpsOnlyOnce(op_confs);
  OF_CUDA_CHECK(cudaFree(workspace));
  OF_CUDA_CHECK(cudaFree(buffer));
  device->DestroyStream(stream);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("CutlassConvTuningWarmupPass", CutlassConvTuningWarmupPass);

}  // namespace oneflow

#endif  // WITH_CUTLASS
