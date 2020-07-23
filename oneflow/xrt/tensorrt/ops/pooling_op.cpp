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
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/xrt/tensorrt/trt_logger.h"

#include "NvInfer.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<nvinfer1::PoolingType pooling_type>
class PoolingOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->SoleInputShape();
    CHECK_GE(in_shape.NumAxes(), 3);
    CHECK_LE(in_shape.NumAxes(), 5);

    const std::string& padding = ctx->Attr<std::string>("padding");
    const auto& pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");

    nvinfer1::ITensor *in = ctx->SoleInput();
    auto *layer =
        ctx->builder()->addPooling(*in, pooling_type, nvinfer1::DimsHW(pool_size[0], pool_size[1]));
    layer->setName(ctx->op_name().c_str());

    layer->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
    if (padding == "same") { layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_LOWER); }
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(MaxPooling2D, PoolingOp<nvinfer1::PoolingType::kMAX>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(AveragePooling2D, PoolingOp<nvinfer1::PoolingType::kAVERAGE>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
