#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

#include "oneflow/xla/of2xla/xla_helpers.h"

namespace oneflow {
namespace mola {

class ReshapeOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");
    Shape shape = ctx->GetAttr<Shape>("shape");

    bool has_batch_dim = ctx->GetAttr<bool>("has_dim0_in_shape");
    if (!has_batch_dim) {
      std::vector<int64_t> dim_vec;
      dim_vec.push_back(in_shape.At(0));
      dim_vec.insert(dim_vec.end(), shape.dim_vec().begin(),
                     shape.dim_vec().end());
      shape = Shape(dim_vec);
    }

    int missing_axes = -1;
    for (int i = 0; i < shape.NumAxes(); ++i) {
      if (shape.At(i) == -1) {
        CHECK_EQ(missing_axes, -1);
        missing_axes = i;
      }
    }
    if (missing_axes >= 0) {
      int64_t missing_dim = -in_shape.Count(0) / shape.Count(0);
      CHECK_GT(missing_dim, 0);
      shape.Set(missing_axes, missing_dim);
    }

    CHECK_EQ(shape.Count(0), in_shape.Count(0));
    ctx->SetOutput("out", Reshape(ctx->Input("in"), shape));
  }
};

REGISTER_XLA_OP_COMPILER(Reshape, ReshapeOp);

}  // namespace mola
}  // namespace oneflow
