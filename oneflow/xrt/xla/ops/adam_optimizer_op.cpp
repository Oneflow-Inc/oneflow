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
#include "oneflow/xrt/xla/ops/optimizer_op.h"

namespace oneflow {
namespace xrt {
namespace mola {

class AdamOptimizerOp : public OptimizerOp {
 private:
  void ApplyUpdate(XlaOpContext *ctx, xla::XlaOp gradient, xla::XlaOp learning_rate) override {
    xla::XlaOp weight = ctx->Input("model_0");
    xla::XlaOp m = ctx->Input("m_0");
    xla::XlaOp v = ctx->Input("v_0");
    double scale_val = ctx->Attr<double>("scale");
    float l1_val = ctx->Attr<float>("l1");
    float l2_val = ctx->Attr<float>("l2");
    float beta1_val = ctx->Attr<float>("beta1");
    float beta2_val = ctx->Attr<float>("beta2");
    float epsilon_val = ctx->Attr<float>("epsilon");
    bool do_bias_correction = ctx->Attr<bool>("do_bias_correction");
    float weight_decay_val = ctx->Attr<float>("weight_decay");
    xla::XlaOp one = One(ctx->builder(), ctx->InputType("m_0"));
    xla::XlaOp lr;
    xla::XlaOp beta1_t;
    xla::XlaOp beta2_t;
    xla::XlaOp scale_by_tensor;
    if (do_bias_correction) {
      beta1_t = ctx->Input("beta1_t_0");
      beta2_t = ctx->Input("beta2_t_0");
      lr = learning_rate * xla::Sqrt(one - beta2_t) / (one - beta1_t);
    } else {
      lr = learning_rate;
    }
    if (ctx->HasInput("scale_by_tensor_0")) { scale_by_tensor = ctx->Input("scale_by_tensor_0"); }
    Shape gradient_shape = ctx->InputShape("model_diff_0");
    if (gradient_shape.NumAxes() > 1) {
      std::vector<long long> bcast_sizes;
      for (int i = 0; i < gradient_shape.NumAxes() - 1; ++i) {
        bcast_sizes.push_back(gradient_shape.At(i));
      }
      lr = xla::Broadcast(lr, bcast_sizes);
      if (ctx->HasInput("scale_by_tensor_0")) {
        scale_by_tensor = xla::Broadcast(scale_by_tensor, bcast_sizes);
      }
    }
    DataType model_dtype = ctx->InputType("model_0");
    DataType gradient_dtype = ctx->InputType("model_diff_0");

    if (model_dtype != gradient_dtype) {
      xla::PrimitiveType data_type = DataTypeToPrimitiveType(model_dtype);
      gradient = xla::ConvertElementType(gradient, data_type);
    }
    if (ctx->HasInput("scale_by_tensor_0")) { gradient = gradient * scale_by_tensor; }
    if (scale_val != 1) {
      xla::XlaOp scale = xla::ScalarLike(gradient, scale_val);
      gradient = gradient * scale;
    }
    if (std::abs(l1_val) != 0) {
      xla::XlaOp l1 = xla::ScalarLike(gradient, l1_val);
      gradient = gradient + l1 * xla::Sign(weight);
    }
    if (std::abs(l2_val) != 0) {
      xla::XlaOp l2 = xla::ScalarLike(gradient, l2_val);
      gradient = gradient + l2 * weight;
    }

    xla::XlaOp beta1 = xla::ScalarLike(m, beta1_val);
    xla::XlaOp beta2 = xla::ScalarLike(v, beta2_val);
    m = beta1 * m + (one - beta1) * gradient;
    v = beta2 * v + (one - beta2) * gradient * gradient;
    xla::XlaOp epsilon = xla::ScalarLike(v, epsilon_val);
    xla::XlaOp weight_decay = xla::ScalarLike(weight, weight_decay_val);

    ctx->SetOutput("m_0", m);
    ctx->SetOutput("v_0", v);
    ctx->SetOutput("model_0", weight - lr * (m / (xla::Sqrt(v) + epsilon) + weight_decay * weight));
    if (do_bias_correction) {
      beta1_t = beta1_t * xla::ScalarLike(beta1_t, beta1_val);
      beta2_t = beta2_t * xla::ScalarLike(beta2_t, beta2_val);
      ctx->SetOutput("beta1_t_0", beta1_t);
      ctx->SetOutput("beta2_t_0", beta2_t);
    }
  }
};

REGISTER_XLA_OP_KERNEL(AdamOptimizer, AdamOptimizerOp)
    .SetIsOptimizerOp(true)
    .SetMutableVariables({"model_0", "m_0", "v_0", "beta1_t_0", "beta2_t_0"})
    .Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
