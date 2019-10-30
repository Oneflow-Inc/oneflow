#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

class ClipGradientOp : public OpKernel {
 public:
  void Compile(OpContext *ctx) override;
};

void ClipGradientOp::Compile(OpContext *ctx) {
  xla::XlaOp instance_num = ctx->Input("instance_num_diff");
  xla::XlaOp gradient = ctx->Input("gradient");
  Shape gradient_shape = ctx->InputShape("gradient");

  xla::XlaOp norm;
  if (ctx->HasAttr("global_norm")) {
    float global_norm_val = ctx->GetAttr<float>("global_norm");
    norm = xla::ScalarLike(gradient, global_norm_val);
  } else {
    // int64_t count = gradient_shape.elem_cnt();
    // xla::XlaOp flat = Reshape(gradient, Shape({count}));
    // norm = xla::Sqrt(xla::Dot(flat, flat)) / instance_num;
    DataType data_type = ctx->InputType("gradient");
    xla::XlaBuilder *builder = ctx->builder();
    std::vector<long long> reduce_dims(gradient_shape.NumAxes());
    std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
    xla::XlaComputation add_func = CreateAddFunc(data_type);
    xla::XlaOp sum = xla::Reduce(gradient * gradient, Zero(builder, data_type),
                                 add_func, reduce_dims);
    norm = xla::Sqrt(sum) / instance_num;
  }

  float clip_norm_val = ctx->GetAttr<float>("clip_norm");
  xla::XlaOp clip_norm = xla::ScalarLike(gradient, clip_norm_val);
  clip_norm = clip_norm / xla::Max(norm, clip_norm);
  if (gradient_shape.NumAxes() > 1) {
    std::vector<long long> broadcast_sizes;
    for (int i = 0; i < gradient_shape.NumAxes() - 1; ++i) {
      broadcast_sizes.push_back(gradient_shape.At(i));
    }
    clip_norm = xla::Broadcast(clip_norm, broadcast_sizes);
  }
  ctx->SetOutput("out", clip_norm * ctx->Input("gradient"));
}

REGISTER_XLA_OP_COMPILER(ClipGradient, ClipGradientOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
