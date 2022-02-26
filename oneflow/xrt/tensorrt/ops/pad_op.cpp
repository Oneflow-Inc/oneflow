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
#include "oneflow/xrt/tensorrt/trt_logger.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class PaddingOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    // TODO(hjchen2): Support not constant 0 padding
    if (ctx->HasAttr("floating_constant_value")) {
      double value = ctx->Attr<double>("floating_constant_value");
      CHECK_EQ(value, 0) << "Only support constant 0 padding";
    }
    if (ctx->HasAttr("integral_constant_value")) {
      int64_t value = ctx->Attr<int64_t>("integral_constant_value");
      CHECK_EQ(value, 0) << "Only support constant 0 padding";
    }

    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
    CHECK_EQ(padding_before.size(), padding_after.size());
    CHECK_EQ(padding_before.size(), 4);
    if (padding_before[0] != 0 || padding_before[1] != 0 || padding_after[0] != 0 || padding_after[1] != 0) {
      UNIMPLEMENTED() << "TensorRT does not support padding batch and channel dimension";
    }
    
    nvinfer1::ITensor* x = ctx->SoleInput();
    nvinfer1::DimsHW prePadding{static_cast<int32_t>(padding_before[2]),
                                static_cast<int32_t>(padding_before[3])};
    nvinfer1::DimsHW postPadding{static_cast<int32_t>(padding_after[2]),
                                 static_cast<int32_t>(padding_after[3])};
    auto* layer =
        ctx->builder()->addPaddingNd(*x, prePadding, postPadding);
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Padding, PaddingOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
