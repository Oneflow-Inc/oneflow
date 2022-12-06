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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/lazy_mode.h"
#include <cutlass/library/handle.h>
#include <cutlass/library/library.h>
#include <cutlass/library/singleton.h>

namespace oneflow {

namespace {

class Conv2dCutlassKernel final : public user_op::OpKernel {
 public:
  Conv2dCutlassKernel() = default;
  ~Conv2dCutlassKernel() override = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return nullptr;
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    auto dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    auto strides = ctx->Attr<std::vector<int32_t>>("strides");

    const int n = in->shape_view().At(0);
    const int h = in->shape_view().At(1);
    const int w = in->shape_view().At(2);
    const int c = in->shape_view().At(3);

    const int k = weight->shape_view().At(0);
    const int r = weight->shape_view().At(1);
    const int s = weight->shape_view().At(2);
    // const int c = in->shape_view().At(3);

    // const int n = out->shape_view().At(0);
    const int p = out->shape_view().At(1);
    const int q = out->shape_view().At(2);
    // const int c = out->shape_view().At(3);

    auto* stream = ctx->stream()->As<ep::CudaStream>();

    cutlass::library::ConvFunctionalKey key(
        cutlass::library::Provider::kCUTLASS, cutlass::library::ConvKind::kFprop,
        cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kF32, cutlass::library::NumericTypeID::kF32);

    cutlass::conv::Conv2dProblemSize problem_size(
        n, h, w, c, k, r, s, p, q, padding_before.at(0), padding_before.at(1), strides.at(0),
        strides.at(1), dilation_rate.at(0), dilation_rate.at(1),
        cutlass::conv::Mode::kCrossCorrelation);
    cutlass::library::Conv2dConfiguration configuraion;
    configuraion.split_k_mode = cutlass::conv::SplitKMode::kNone;
    configuraion.problem_size = problem_size;
    configuraion.stride_a = {h * w * c, w * c, c, 1};
    configuraion.stride_b = {r * s * c, s * c, c, 1};
    // configuraion.stride_c = {0, 0, 0, 1};
    configuraion.stride_c = {p * q * k, q * k, k, 1};

    cutlass::library::ConvArguments arguments;
    arguments.A = in->dptr();
    arguments.B = weight->dptr();
    arguments.reordered_B = nullptr;
    arguments.C = nullptr;
    arguments.D = out->mut_dptr();

    float alpha = 1;
    float beta = 0;
    arguments.alpha = &alpha;
    arguments.beta = &beta;
    arguments.pointer_mode = cutlass::library::ScalarPointerMode::kHost;

    const auto& operations_map_it =
        cutlass::library::Singleton::get().operation_table.conv2d_operations.find(key);
    if (operations_map_it
        == cutlass::library::Singleton::get().operation_table.conv2d_operations.cend()) {
      LOG(ERROR) << "op not found";
      return;
    }
    const cutlass::library::ConvOperationVectorMap& operations_map = operations_map_it->second;
    for (const auto& pair : operations_map) {
      LOG(ERROR) << pair.first.compute_capability << " " << pair.first.iterator_algorithm << " "
                 << pair.second.size();
      for (auto operation : pair.second) {
        auto status = operation->can_implement(&configuraion, &arguments);
        if (status != cutlass::Status::kSuccess) { continue; }
        const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
        const size_t device_workspace_size = operation->get_device_workspace_size(&configuraion);
        if (device_workspace_size > tmp_buffer->shape_view().elem_cnt()) { continue; }
        std::vector<uint8_t> host_workspace(host_workspace_size, 0);
        auto init_status = operation->initialize(&configuraion, host_workspace.data(),
                                                 tmp_buffer->mut_dptr(), stream->cuda_stream());
        LOG(ERROR) << "init " << init_status;
        auto run_status = operation->run(&arguments, host_workspace.data(), tmp_buffer->mut_dptr(),
                                         stream->cuda_stream());
        LOG(ERROR) << "run " << run_status;
      }
    }
  }
};

REGISTER_USER_KERNEL("conv2d")
    .SetCreateFn<Conv2dCutlassKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobAttr<std::string>("data_format") == "channels_last")
                     && (user_op::HobDataType("in", 0) == DataType::kFloat16))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t { return 1 << 30; })
    .SetPriority(user_op::kKernelPriorityExperimental);

}  // namespace

}  // namespace oneflow
