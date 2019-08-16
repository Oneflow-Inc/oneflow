#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

#include "oneflow/xla/of2xla/xla_helpers.h"

namespace oneflow {
namespace mola {

class LayerNormOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override;

 private:
  xla::XlaOp BatchNormTraining(const xla::XlaOp &input, const xla::XlaOp &scale,
                               const xla::XlaOp &shift, double epsilon) {
    // Feature index is 1 for NCHW
    int feature_index = 1;
    return xla::BatchNormTraining(input, scale, shift, epsilon, feature_index);
  }
};

void LayerNormOp::Compile(XlaOpContext *ctx) {
  DataType data_type = ctx->InputType("in");
  // input layout [N, C, H, W]
  Shape input_shape = ctx->InputShape("in");

  int begin_norm_axis = ctx->GetAttr<int64_t>("begin_norm_axis");
  int begin_params_axis = ctx->GetAttr<int64_t>("begin_params_axis");
  CHECK_LT(begin_norm_axis, input_shape.NumAxes());
  CHECK_LT(begin_params_axis, input_shape.NumAxes());
  while (begin_norm_axis < 0) {
    begin_norm_axis += input_shape.NumAxes();
  }

  int64_t batch_dims = input_shape.Count(0, begin_norm_axis);
  int64_t norm_dims = input_shape.Count(begin_norm_axis);
  Shape bn_shape = Shape({1, batch_dims, 1, norm_dims});
  Shape scale_shape = Shape({batch_dims});
  // Ones and zeros layout [N]
  xla::XlaOp ones = Ones(ctx->builder(), scale_shape, data_type);
  xla::XlaOp zeros = Zeros(ctx->builder(), scale_shape, data_type);
  // input layout [1, N, 1, CHW]
  xla::XlaOp input = Reshape(ctx->Input("in"), bn_shape);

  double epsilon = ctx->GetAttr<double>("epsilon");
  // BatchNorm
  xla::XlaOp norm_output = BatchNormTraining(input, ones, zeros, epsilon);
  // Set outputs, scale and shift
  xla::XlaOp output = xla::GetTupleElement(norm_output, 0);
  xla::XlaOp mean = xla::GetTupleElement(norm_output, 1);
  xla::XlaOp variance = xla::GetTupleElement(norm_output, 2);
  xla::XlaOp inv_variance =
      xla::Rsqrt(xla::Add(variance, xla::ScalarLike(variance, epsilon)));

  Shape mean_shape = SliceShape(input_shape, 0, begin_norm_axis);
  ctx->SetOutput("mean", Reshape(mean, mean_shape));
  ctx->SetOutput("inv_variance", Reshape(inv_variance, mean_shape));

  if (ctx->GetAttr<bool>("scale")) {
    ctx->SetOutput("normalized", Reshape(output, input_shape));
  }

  Shape gamma_shape = Shape({norm_dims});
  // output = Reshape(output, Shape({batch_dims, norm_dims}));
  if (ctx->GetAttr<bool>("scale")) {
    CHECK_EQ(gamma_shape, ctx->InputShape("gamma"));
    output = xla::Mul(output, ctx->Input("gamma"), {3}/*broadcast dim*/);
  }
  if (ctx->GetAttr<bool>("center")) {
    CHECK_EQ(gamma_shape, ctx->InputShape("beta"));
    output = xla::Add(output, ctx->Input("beta"), {3}/*broadcast dim*/);
  }
  ctx->SetOutput("out", Reshape(output, input_shape));
}

REGISTER_XLA_OP_COMPILER(LayerNorm, LayerNormOp);

}  // namespace mola
}  // namespace oneflow
