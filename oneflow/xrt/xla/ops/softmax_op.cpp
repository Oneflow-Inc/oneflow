#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

class SoftmaxOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override;
};

void SoftmaxOp::Compile(XlaOpContext *ctx) {
  xla::XlaBuilder *builder = ctx->builder();
  Shape input_shape = ctx->InputShape("in");

  int axis = ctx->GetAttr<int32_t>("axis");
  if (axis < 0) { axis += input_shape.NumAxes(); }
  std::vector<long long> batch_dims(input_shape.NumAxes() - 1);
  // std::iota(batch_dims.begin(), batch_dims.end(), 0);
  for (int i = 0; i < axis; ++i) { batch_dims[i] = i; }
  for (int i = axis; i < input_shape.NumAxes() - 1; ++i) { batch_dims[i] = i + 1; }

  DataType data_type = ctx->InputType("in");
  xla::XlaOp input = ctx->Input("in");
  xla::XlaComputation max_func = CreateMaxFunc(data_type);
  xla::XlaOp logits_max = xla::Reduce(input, MinValue(builder, data_type), max_func, {axis});
  // y = exp(x - max)
  xla::XlaOp y = xla::Exp(xla::Sub(input, logits_max, batch_dims));

  xla::XlaComputation add_func = CreateAddFunc(data_type);
  // TODO(hjchen2) Accumulate by float if use bfloat16
  xla::XlaOp sum = xla::Reduce(y, Zero(builder, data_type), add_func, {axis});
  // exp(x - max) / sum(exp(x - max))
  ctx->SetOutput("out", xla::Div(y, sum, batch_dims));
}

REGISTER_XLA_OP_KERNEL(Softmax, SoftmaxOp).Finalize();

class SoftmaxGradOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override;
};

// softmax gradient formula:
// dx = y * (dy - sum(dy * y))
void SoftmaxGradOp::Compile(XlaOpContext *ctx) {
  Shape y_shape = ctx->InputShape("y");

  int axis = ctx->GetAttr<int32_t>("axis");
  if (axis < 0) { axis += y_shape.NumAxes(); }

  std::vector<long long> batch_dims(y_shape.NumAxes() - 1);
  // std::iota(batch_dims.begin(), batch_dims.end(), 0);
  for (int i = 0; i < axis; ++i) { batch_dims[i] = i; }
  for (int i = axis; i < y_shape.NumAxes() - 1; ++i) { batch_dims[i] = i + 1; }

  xla::XlaOp y = ctx->Input("y");
  xla::XlaOp dy = ctx->Input("dy");
  DataType data_type = ctx->InputType("y");
  xla::XlaBuilder *builder = ctx->builder();

  xla::XlaComputation add_func = CreateAddFunc(data_type);
  xla::XlaOp sum = xla::Reduce(y * dy, Zero(builder, data_type), add_func, {axis});
  ctx->SetOutput("dx", y * xla::Sub(dy, sum, batch_dims));
}

REGISTER_XLA_OP_KERNEL(SoftmaxGrad, SoftmaxGradOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
