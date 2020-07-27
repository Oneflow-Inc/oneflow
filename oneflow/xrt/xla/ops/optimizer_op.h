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
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp gradient = ctx->Input("model_diff");
    xla::XlaOp learning_rate = ctx->Input("learning_rate");
    ApplyUpdate(ctx, gradient, learning_rate);
  }

 private:
  virtual void ApplyUpdate(XlaOpContext *ctx, xla::XlaOp gradient, xla::XlaOp learning_rate) = 0;

  xla::XlaOp ClipGradient(XlaOpContext *ctx, const xla::XlaOp &gradient,
                          const ClipConf &clip_conf) {
    DataType data_type = ctx->InputType("model_diff");
    Shape gradient_shape = ctx->InputShape("model_diff");
    xla::XlaOp norm;
    if (clip_conf.clip_by_global_norm().has_global_norm()) {
      float global_norm_val = clip_conf.clip_by_global_norm().global_norm();
      norm = xla::ScalarLike(gradient, global_norm_val);
    } else {
      // int64_t count = gradient_shape.elem_cnt();
      // xla::XlaOp flat = Reshape(gradient, Shape({count}));
      // norm = xla::Sqrt(xla::Dot(flat, flat));
      xla::XlaBuilder *builder = ctx->builder();
      std::vector<long long> reduce_dims(gradient_shape.NumAxes());
      std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
      xla::XlaComputation add_func = CreateAddFunc(data_type);
      xla::XlaOp sum =
          xla::Reduce(gradient * gradient, Zero(builder, data_type), add_func, reduce_dims);
      norm = xla::Sqrt(sum);
    }

    float clip_norm_val = clip_conf.clip_by_global_norm().clip_norm();
    xla::XlaOp clip_norm = xla::ScalarLike(gradient, clip_norm_val);
    clip_norm = clip_norm / xla::Max(norm, clip_norm);
    if (gradient_shape.NumAxes() > 1) {
      std::vector<long long> broadcast_sizes;
      for (int i = 0; i < gradient_shape.NumAxes() - 1; ++i) {
        broadcast_sizes.push_back(gradient_shape.At(i));
      }
      clip_norm = xla::Broadcast(clip_norm, broadcast_sizes);
    }

    return clip_norm * gradient;
  }
};

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_OPS_OPTIMIZER_OP_H_
