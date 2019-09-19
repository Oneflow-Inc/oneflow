#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

xla::XlaOp GenericGather(const xla::XlaOp &input, const xla::XlaOp &indices,
                         const Shape &input_shape, const Shape &indices_shape,
                         int64_t axis) {
  int64_t index_vector_dim = indices_shape.NumAxes();
  xla::GatherDimensionNumbers dim_numbers;
  std::vector<long long> slice_sizes(input_shape.NumAxes());

  for (int64_t i = 0; i < input_shape.NumAxes(); ++i) {
    int64_t window_bound;
    if (i == axis) {
      dim_numbers.add_collapsed_slice_dims(i);
      window_bound = 1;
    } else {
      window_bound = input_shape.At(i);
      if (i < axis) {
        dim_numbers.add_offset_dims(i);
      } else if (i >= (axis + 1)) {
        dim_numbers.add_offset_dims(i + index_vector_dim - 1);
      }
    }

    slice_sizes[i] = window_bound;
  }
  
  dim_numbers.set_index_vector_dim(index_vector_dim);
  dim_numbers.add_start_index_map(axis);

  return xla::Gather(input, indices, dim_numbers, slice_sizes);
}

class GatherOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape input_shape = ctx->InputShape("in");
    Shape indices_shape = ctx->InputShape("indices");
    CHECK_GT(input_shape.NumAxes(), 0);
    CHECK_GT(indices_shape.NumAxes(), 0);
    CHECK_LE(indices_shape.NumAxes(), input_shape.NumAxes());

    xla::XlaOp input = ctx->Input("in");
    xla::XlaOp indices = ctx->Input("indices");
    int axis = GatherAxis(ctx);
    int batch_dims = GatherBatchDims(ctx);

    xla::XlaOp output;
    if (batch_dims > 0) {
      output = xla::TorchIndexSelect(input, indices, axis, batch_dims);
    } else {
      output = GenericGather(input, indices, input_shape, indices_shape,
                             axis);
    }
    ctx->SetOutput("out", output);
  }

  virtual int GatherAxis(XlaOpContext *ctx) const {
    return ctx->GetAttr<int64_t>("axis");
  }
  virtual int GatherBatchDims(XlaOpContext *ctx) const {
    return 0;
  }
};

class BatchGatherOp : public GatherOp {
 public:
  int GatherAxis(XlaOpContext *ctx) const override {
    Shape indices_shape = ctx->InputShape("indices");
    return indices_shape.NumAxes() - 1;
  }
  int GatherBatchDims(XlaOpContext *ctx) const override {
    Shape indices_shape = ctx->InputShape("indices");
    return indices_shape.NumAxes() - 1;
  }
};

REGISTER_XLA_OP_COMPILER(Gather, GatherOp);
REGISTER_XLA_OP_COMPILER(BatchGather, BatchGatherOp);

}  // namespace mola
}  // namespace oneflow
