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
#ifndef ONEFLOW_XRT_XLA_OPS_OPTIMIZER_OP_H_
#define ONEFLOW_XRT_XLA_OPS_OPTIMIZER_OP_H_

#include "oneflow/core/operator/op_conf.pb.h"

#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

class OptimizerOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext* ctx) override {
    xla::XlaOp gradient = ctx->Input("model_diff_0");
    xla::XlaOp learning_rate = ctx->Input("learning_rate_0");
    ApplyUpdate(ctx, gradient, learning_rate);
  }

 private:
  virtual void ApplyUpdate(XlaOpContext* ctx, xla::XlaOp gradient, xla::XlaOp learning_rate) = 0;
};

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_OPS_OPTIMIZER_OP_H_
