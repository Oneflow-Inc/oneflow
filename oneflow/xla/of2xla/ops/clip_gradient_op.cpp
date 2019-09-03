#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

#include "oneflow/xla/of2xla/xla_helpers.h"

namespace oneflow {
namespace mola {

class ClipGradientOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override;
};

void ClipGradientOp::Compile(XlaOpContext *ctx) {
  // TODO(hjchen2)
  xla::XlaOp gradient = ctx->Input("gradient");
  xla::XlaOp instance_num = ctx->Input("instance_num_diff");

  xla::XlaOp norm;
  if (ctx->HasAttr("global_norm")) {
    float global_norm_val = ctx->GetAttr<float>("global_norm");
    norm = xla::ScalarLike(gradient, global_norm_val);
  } else {
    Shape gradient_shape = ctx->InputShape("gradient");
    int64_t count = gradient_shape.elem_cnt();
    xla::XlaOp flat = Reshape(gradient, Shape({count}));
    norm = xla::Sqrt(xla::Dot(flat, flat)) / instance_num;
  }
  
  float clip_norm_val = ctx->GetAttr<float>("clip_norm");
  xla::XlaOp clip_norm = xla::ScalarLike(gradient, clip_norm_val);
  ctx->SetOutput("out", clip_norm / xla::Max(norm, clip_norm) *
                        ctx->Input("gradient"));
}

REGISTER_XLA_OP_COMPILER(ClipGradient, ClipGradientOp);

}  // namespace mola
}  // namespace oneflow
