#include "oneflow/xrt/xla/ops/optimizer_op.h"

namespace oneflow {
namespace xrt {
namespace mola {

class AdamOptimizerOp : public OptimizerOp {
 private:
  void ApplyUpdate(XlaOpContext *ctx, xla::XlaOp gradient, xla::XlaOp learning_rate) override {
    xla::XlaOp weight = ctx->Input("model");
    xla::XlaOp m = ctx->Input("m");
    xla::XlaOp v = ctx->Input("v");
    Shape gradient_shape = ctx->InputShape("model_diff");
    if (gradient_shape.NumAxes() > 1) {
      std::vector<long long> bcast_sizes;
      for (int i = 0; i < gradient_shape.NumAxes() - 1; ++i) {
        bcast_sizes.push_back(gradient_shape.At(i));
      }
      learning_rate = xla::Broadcast(learning_rate, bcast_sizes);
    }

    NormalModelUpdateOpUserConf *user_conf =
        dynamic_cast<NormalModelUpdateOpUserConf *>(ctx->GetAttr<PbMessage *>("user_conf"));
    CHECK(user_conf) << "Can not get message `user_conf`.";
    if (user_conf->has_adam_conf()) {
      xla::XlaOp one = One(ctx->builder(), ctx->InputType("m"));
      float beta1_val = user_conf->adam_conf().beta1();
      float beta2_val = user_conf->adam_conf().beta2();
      xla::XlaOp beta1 = xla::ScalarLike(m, beta1_val);
      xla::XlaOp beta2 = xla::ScalarLike(v, beta2_val);
      m = beta1 * m + (one - beta1) * gradient;
      v = beta2 * v + (one - beta2) * gradient * gradient;
      ctx->SetOutput("m", m);
      ctx->SetOutput("v", v);
      // ctx->SetOutput("out_m", m);
      // ctx->SetOutput("out_v", v);

      float epsilon_val = user_conf->adam_conf().epsilon();
      xla::XlaOp epsilon = xla::ScalarLike(v, epsilon_val);
      gradient = m / (xla::Sqrt(v) + epsilon);
    }

    ctx->SetOutput("model", weight - learning_rate * gradient);
    // ctx->SetOutput("out", weight - learning_rate * gradient);
  }
};

REGISTER_XLA_OP_KERNEL(AdamOptimizer, AdamOptimizerOp)
    .SetIsOptimizerOp(true)
    .SetMutableVariables({"model", "m", "v"})
    .Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
