#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

class LayerNormOp : public XlaOpKernel {
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
  // input layout [N, C, H, W]
  Shape input_shape = ctx->InputShape("in");

  int begin_norm_axis = ctx->GetAttr<int64_t>("begin_norm_axis");
  int begin_params_axis = ctx->GetAttr<int64_t>("begin_params_axis");
  CHECK_LT(begin_norm_axis, input_shape.NumAxes());
  CHECK_LT(begin_params_axis, input_shape.NumAxes());
  while (begin_norm_axis < 0) { begin_norm_axis += input_shape.NumAxes(); }

  DataType data_type = ctx->InputType("in");
  if (ctx->HasOutput("mean")) { data_type = ctx->OutputType("mean"); }
  int64_t batch_dims = input_shape.Count(0, begin_norm_axis);
  int64_t norm_dims = input_shape.Count(begin_norm_axis);
  Shape bn_shape = Shape({1, batch_dims, 1, norm_dims});
  Shape scale_shape = Shape({batch_dims});
  // Ones and zeros layout [N]
  xla::XlaOp ones = Ones(ctx->builder(), scale_shape, data_type);
  xla::XlaOp zeros = Zeros(ctx->builder(), scale_shape, data_type);
  // input layout [1, N, 1, CHW]
  xla::XlaOp input = Reshape(ctx->Input("in"), bn_shape);

  // FP16 batch normalization (cudnn style) has not been supported by XLA.
  if (ctx->InputType("in") != data_type) {
    input = xla::ConvertElementType(input, DataTypeToPrimitiveType(data_type));
  }

  double epsilon = ctx->GetAttr<double>("epsilon");
  // BatchNorm
  xla::XlaOp norm_output = BatchNormTraining(input, ones, zeros, epsilon);
  // Set outputs, scale and shift
  xla::XlaOp output = xla::GetTupleElement(norm_output, 0);
  xla::XlaOp mean = xla::GetTupleElement(norm_output, 1);
  xla::XlaOp variance = xla::GetTupleElement(norm_output, 2);
  xla::XlaOp inv_variance = xla::Rsqrt(xla::Add(variance, xla::ScalarLike(variance, epsilon)));

  Shape mean_shape = SliceShape(input_shape, 0, begin_norm_axis);
  if (ctx->HasOutput("mean")) { ctx->SetOutput("mean", Reshape(mean, mean_shape)); }
  if (ctx->HasOutput("inv_variance")) {
    ctx->SetOutput("inv_variance", Reshape(inv_variance, mean_shape));
  }

  if (ctx->OutputType("out") != data_type) {
    DataType output_type = ctx->OutputType("out");
    output = xla::ConvertElementType(output, DataTypeToPrimitiveType(output_type));
  }

  if (ctx->GetAttr<bool>("scale") && ctx->HasOutput("normalized")) {
    ctx->SetOutput("normalized", Reshape(output, input_shape));
  }

  Shape gamma_shape = Shape({norm_dims});
  // output = Reshape(output, Shape({batch_dims, norm_dims}));
  if (ctx->GetAttr<bool>("scale")) {
    CHECK_EQ(gamma_shape, ctx->InputShape("gamma"));
    output = xla::Mul(output, ctx->Input("gamma"), {3} /*broadcast dim*/);
  }
  if (ctx->GetAttr<bool>("center")) {
    CHECK_EQ(gamma_shape, ctx->InputShape("beta"));
    output = xla::Add(output, ctx->Input("beta"), {3} /*broadcast dim*/);
  }

  ctx->SetOutput("out", Reshape(output, input_shape));
}

class LayerNormGradOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override;

 private:
  xla::XlaOp BatchNormGrad(const xla::XlaOp &activations, const xla::XlaOp &scale,
                           const xla::XlaOp &mean, const xla::XlaOp &variance,
                           const xla::XlaOp &grad, double epsilon) {
    // Feature index is 1 for NCHW
    int feature_index = 1;
    return xla::BatchNormGrad(activations, scale, mean, variance, grad, epsilon, feature_index);
  }
};

void LayerNormGradOp::Compile(XlaOpContext *ctx) {
  xla::XlaOp output_grad = ctx->Input("dy");
  xla::XlaOp activation = ctx->Input("x");
  xla::XlaOp mean = ctx->Input("mean");
  xla::XlaOp inv_variance = ctx->Input("inv_variance");

  Shape activation_shape = ctx->InputShape("x");
  int begin_norm_axis = ctx->GetAttr<int64_t>("begin_norm_axis");
  CHECK_LT(begin_norm_axis, activation_shape.NumAxes());
  while (begin_norm_axis < 0) { begin_norm_axis += activation_shape.NumAxes(); }

  int64_t batch_dims = activation_shape.Count(0, begin_norm_axis);
  int64_t norm_dims = activation_shape.Count(begin_norm_axis);
  Shape bn_shape = Shape({1, batch_dims, 1, norm_dims});
  Shape scale_shape = Shape({batch_dims});

  double epsilon = ctx->GetAttr<double>("epsilon");
  xla::XlaOp ones = xla::ScalarLike(inv_variance, 1.0f);
  xla::XlaOp variance =
      xla::Sub(ones / (inv_variance * inv_variance), xla::ScalarLike(inv_variance, epsilon));

  activation = Reshape(activation, bn_shape);
  mean = Reshape(mean, scale_shape);
  variance = Reshape(variance, scale_shape);
  output_grad = Reshape(output_grad, bn_shape);

  if (ctx->InputType("mean") != ctx->InputType("x")) {
    DataType data_type = ctx->InputType("mean");
    activation = xla::ConvertElementType(activation, DataTypeToPrimitiveType(data_type));
    output_grad = xla::ConvertElementType(output_grad, DataTypeToPrimitiveType(data_type));
  }

  auto output = BatchNormGrad(activation, xla::Broadcast(ones, {batch_dims}), mean, variance,
                              output_grad, epsilon);
  xla::XlaOp activation_grad = xla::GetTupleElement(output, 0);

  if (ctx->InputType("mean") != ctx->InputType("x")) {
    DataType data_type = ctx->InputType("x");
    activation_grad = xla::ConvertElementType(activation_grad, DataTypeToPrimitiveType(data_type));
  }
  ctx->SetOutput("dx", Reshape(activation_grad, activation_shape));
}

class LayerNormParamGradOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override;
};

void LayerNormParamGradOp::Compile(XlaOpContext *ctx) {
  xla::XlaOp output_grad = ctx->Input("dy");
  Shape output_shape = ctx->InputShape("dy");

  int begin_params_axis = ctx->GetAttr<int64_t>("begin_params_axis");
  while (begin_params_axis < 0) { begin_params_axis += output_shape.NumAxes(); }
  std::vector<long long> batch_dims(begin_params_axis);
  std::iota(batch_dims.begin(), batch_dims.end(), 0);
  int norm_dims_size = output_shape.NumAxes() - begin_params_axis;
  std::vector<long long> norm_dims(norm_dims_size);
  std::iota(norm_dims.begin(), norm_dims.end(), begin_params_axis);

  xla::XlaBuilder *builder = ctx->builder();
  DataType data_type = ctx->InputType("dy");
  xla::XlaComputation add_func = CreateAddFunc(data_type);
  if (ctx->HasAttr("beta_diff")) {
    xla::XlaOp beta_grad = xla::Reduce(output_grad, Zero(builder, data_type), add_func, batch_dims);
    ctx->SetOutput("beta_diff", beta_grad);
  }
  if (ctx->HasAttr("gamma_diff")) {
    xla::XlaOp normalized = ctx->Input("normalized");
    xla::XlaOp gamma_grad = normalized * output_grad;
    gamma_grad = xla::Reduce(gamma_grad, Zero(builder, data_type), add_func, batch_dims);
    ctx->SetOutput("gamma_diff", gamma_grad);
  }
  if (ctx->HasAttr("normalized_diff")) {
    xla::XlaOp normalized_grad;
    if (ctx->HasAttr("gamma")) {
      xla::XlaOp gamma = ctx->Input("gamma");
      normalized_grad = xla::Mul(output_grad, gamma, norm_dims);
    } else {
      normalized_grad = output_grad;
    }
    ctx->SetOutput("normalized_diff", normalized_grad);
  }
}

REGISTER_XLA_OP_KERNEL(LayerNorm, LayerNormOp).Finalize();
REGISTER_XLA_OP_KERNEL(LayerNormParamGrad, LayerNormParamGradOp).Finalize();
REGISTER_XLA_OP_KERNEL(LayerNormGrad, LayerNormGradOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
