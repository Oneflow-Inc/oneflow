#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"
#include "oneflow/xla/of2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/lib/scatter.h"
namespace oneflow {
namespace mola {

// actually ScatterNdOp
class GatherGradOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    DataType dtype = ctx->InputType("out_diff");
    Shape indices_shape = ctx->InputShape("indices");
    Shape update_shape = ctx->InputShape("out_diff");
    int64_t gather_buffer = ctx->GetAttr<int64_t>("gather_dim_size");
    Shape buffer_shape = Shape({gather_buffer});
    xla::XlaBuilder* builder = ctx->builder();
    auto buffer = xla::Broadcast(Zero(builder, dtype), {buffer_shape.NumAxes()});
    auto indices = ctx->Input("indices");
    auto updates = ctx->Input("out_diff");
    auto result = tensorflow::XlaScatter(buffer, updates, indices, true, Combine, builder);
    ctx->SetOutput("in_diff", result.ValueOrDie());
  }
  private:
   static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y, xla::XlaBuilder* builder ) {
     return xla::Add(x,y);
   }
};

REGISTER_XLA_OP_COMPILER(GatherGrad, GatherGradOp);

xla::XlaOp  GenericGatherGrad(
    XlaOpContext* ctx, 
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp, xla::XlaBuilder*)>& combiner) {
    Shape buffer_shape = Shape({ctx->GetAttr<int64_t>("gather_dim_size")});
    Shape indices_shape = ctx->InputShape("indices");
    Shape updates_shape = ctx->InputShape("out_diff");

    xla::XlaBuilder* builder = ctx->builder();
    auto buffer = ctx->Input("gather_dim_size");
    auto indices = ctx->Input("indices");
    auto updates = ctx->Input ("out_diff");
    auto result = tensorflow::XlaScatter(buffer, updates, indices, true, combiner, builder);
    ctx->SetOutput("in_diff", result.ValueOrDie());
    }
//actually scatterupdate
//class GatherGradOp : public XlaOpCompiler {
// public:
//  void Compile(XlaOpContext ctx) {
//    GenericGatherGrad(ctx, [](xla::XlaOp, xla::XlaOp y, xla::XlaBuilder*) {
//      return y;
//    })
//  }
//};

class BatchGatherGradOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext* ctx) override {
    GenericGatherGrad(
      ctx, [](xla::XlaOp, xla::XlaOp y, xla::XlaBuilder*) { return y; });
  }
};

REGISTER_XLA_OP_COMPILER(BatchGatherGrad, BatchGatherGradOp);

}  // namespace mola
}  // namespace oneflow
