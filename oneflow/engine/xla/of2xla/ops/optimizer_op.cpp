#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"

#include "oneflow/engine/xla/of2xla/xla_helpers.h"

namespace oneflow {
namespace mla {

class AdamOptimizerOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override;
};

void AdamOptimizerOp::Compile(XlaOpContext *ctx) {
  xla::XlaOp instance_num = ctx->Input("instance_num_diff");
  xla::XlaOp learning_rate = ctx->Input("learning_rate");
  xla::XlaOp weight = ctx->Input("weight");
  xla::XlaOp m = ctx->Input("m");
  xla::XlaOp v = ctx->Input("v");
  Shape gradient_shape = ctx->InputShape("gradient");
  if (gradient_shape.NumAxes() > 1) {
    std::vector<long long> bcast_sizes;
    for (int i = 0; i < gradient_shape.NumAxes() - 1; ++i) {
      bcast_sizes.push_back(gradient_shape.At(i));
    }
    instance_num = xla::Broadcast(instance_num, bcast_sizes);
    learning_rate = xla::Broadcast(learning_rate, bcast_sizes);
  }
  xla::XlaOp gradient = ctx->Input("gradient") / instance_num;

  xla::XlaOp one = One(ctx->builder(), ctx->InputType("m"));
  float beta1_val = ctx->GetAttr<float>("beta1");
  float beta2_val = ctx->GetAttr<float>("beta2");
  xla::XlaOp beta1 = xla::ScalarLike(m, beta1_val);
  xla::XlaOp beta2 = xla::ScalarLike(v, beta2_val);
  m = beta1 * m + (one - beta1) * gradient;
  v = beta2 * v + (one - beta2) * gradient * gradient;
  ctx->SetOutput("m", m);
  ctx->SetOutput("v", v);
  // ctx->SetOutput("out_m", m);
  // ctx->SetOutput("out_v", v);

  float epsilon_val = ctx->GetAttr<float>("epsilon");
  xla::XlaOp epsilon = xla::ScalarLike(v,  epsilon_val);
  gradient = m / (xla::Sqrt(v) + epsilon);

  float l1_val = ctx->GetAttr<float>("l1");
  float l2_val = ctx->GetAttr<float>("l2");
  if (std::abs(l1_val) > 1e-6) {
    xla::XlaOp l1 = xla::ScalarLike(gradient, l1_val); 
    gradient = gradient + l1 * xla::Sign(weight);
  }
  if (std::abs(l2_val) > 1e-6) {
    xla::XlaOp l2 = xla::ScalarLike(gradient, l2_val);
    gradient = gradient + l2 * weight;
  }
  ctx->SetOutput("weight", weight - learning_rate * gradient);
  // ctx->SetOutput("out", weight - learning_rate * gradient);
}

REGISTER_XLA_OP_COMPILER(AdamOptimizer, AdamOptimizerOp)
    .MutableVariables({"weight", "m", "v"});

}  // namespace mla
}  // namespace oneflow
