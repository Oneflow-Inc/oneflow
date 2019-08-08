#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

class BatchGatherOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape shape_in = ctx->InputShape("in");
    Shape shape_indices = ctx->InputShape("indices");
    CHECK_GT(shape_in.NumAxes(),0);
    CHECK_GT(shape_indices.NumAxes(),0);


    std::vector<int64_t>& in_dim_vec = shape_in.dim_vec();
    std::vector<int64_t>& indices_dim_vec = shape_indices.dim_vec();
    CHECK_LE(shape_indices.size(), in_dim_vec.size());
    xla::XlaBuilder* builder = ctx->builder();  
    xla::XlaOp gather; 
    ctx->SetOutput("out", xla::Gather(in, shape_in, indices, indices_shape, /*axis=*/0, 
      /*indices_are_nd-*/true, ctx->InputShape("in"), ctx->InpurShape("indices"), builder, &gather));
  }
};

REGISTER_XLA_OP_COMPILER(BatchGather, BatchGatherOp);

}  // namespace mola
}  // namespace oneflow
