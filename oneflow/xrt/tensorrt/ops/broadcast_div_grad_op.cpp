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
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/xrt/tensorrt/plugin/broadcast_div_grad_plugin.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class BroadcastDivGradOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    std::vector<nvinfer1::ITensor*> inputs(3);
    inputs[0] = ctx->Input("y_0");
    inputs[1] = ctx->Input("z_0");
    inputs[2] = ctx->Input("dz_0");
    BroadcastDivGradPlugin plugin(ctx->op_name());
    auto* layer = ctx->builder()->addPluginV2(inputs.data(), 3, plugin);
    ctx->SetOutput("dy_0", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(BcastDivGrad, BroadcastDivGradOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
