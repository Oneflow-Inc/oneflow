#include "oneflow/xrt/xla/op_context.h"
#include "oneflow/xrt/xla/ops/op_compiler.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

class SoftmaxOp : public OpCompiler {
 public:
  void Compile(OpContext *ctx) override;
};

void SoftmaxOp::Compile(OpContext *ctx) {
  xla::XlaBuilder *builder = ctx->builder();
  Shape input_shape = ctx->InputShape("in");
  std::vector<long long> batch_dims(input_shape.NumAxes() - 1);
  std::iota(batch_dims.begin(), batch_dims.end(), 0);

  DataType data_type = ctx->InputType("in");
  xla::XlaOp input = ctx->Input("in");
  xla::XlaComputation max_func = CreateMaxFunc(data_type);
  xla::XlaOp logits_max = xla::Reduce(input, MinValue(builder, data_type),
                                      max_func, {input_shape.NumAxes() - 1});
  // y = exp(x - max)
  xla::XlaOp y = xla::Exp(xla::Sub(input, logits_max, batch_dims));

  xla::XlaComputation add_func = CreateAddFunc(data_type);
  // TODO(hjchen2) Accumulate by float if use float16 or bfloat16
  xla::XlaOp sum = xla::Reduce(y, Zero(builder, data_type), add_func,
                               {input_shape.NumAxes() - 1});
  // exp(x - max) / sum(exp(x - max))
  ctx->SetOutput("out", xla::Div(y, sum, batch_dims));
}

REGISTER_XLA_OP_COMPILER(Softmax, SoftmaxOp).Finalize();

class SoftmaxGradOp : public OpCompiler {
 public:
  void Compile(OpContext *ctx) override;
};

// softmax gradient formula:
// dx = y * (dy - sum(dy * y))
void SoftmaxGradOp::Compile(OpContext *ctx) {
  Shape y_shape = ctx->InputShape("y");
  std::vector<long long> batch_dims(y_shape.NumAxes() - 1);
  std::iota(batch_dims.begin(), batch_dims.end(), 0);

  xla::XlaOp y = ctx->Input("y");
  xla::XlaOp dy = ctx->Input("dy");
  DataType data_type = ctx->InputType("y");
  xla::XlaBuilder *builder = ctx->builder();

  xla::XlaComputation add_func = CreateAddFunc(data_type);
  xla::XlaOp sum = xla::Reduce(y * dy, Zero(builder, data_type), add_func,
                               {y_shape.NumAxes() - 1});
  ctx->SetOutput("dx", y * xla::Sub(dy, sum, batch_dims));
}

REGISTER_XLA_OP_COMPILER(SoftmaxGrad, SoftmaxGradOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
