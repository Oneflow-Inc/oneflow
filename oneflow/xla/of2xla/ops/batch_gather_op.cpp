#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"
#include "absl/types/optional.h"

namespace oneflow {
namespace mola {

class BatchGatherOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape shape_in = ctx->InputShape("in");
    Shape shape_indices = ctx->InputShape("indices");
    CHECK_GT(shape_in.NumAxes(),0);
    CHECK_GT(shape_indices.NumAxes(),0);

    xla::XlaOp in = ctx->Input("in");
    xla::XlaOp indices = ctx->Input("indices");
    std::vector<int64_t> in_dim_vec = shape_in.dim_vec();
    std::vector<int64_t> indices_dim_vec = shape_indices.dim_vec();
    CHECK_LE(indices_dim_vec.size(), in_dim_vec.size());
    int64_t num_index_dims = shape_indices.At(1);
    xla::GatherDimensionNumbers dim_numbers;
    std::vector<long long> slice_sizes;
    int64_t axis = 0;
    slice_sizes.reserve(shape_in.NumAxes());
    for (int64_t i = 0; i < shape_in.NumAxes(); i++) {
      int64_t window_bound;
      if (axis <= i && i < (axis + num_index_dims)) {
        dim_numbers.add_collapsed_slice_dims(i);
        window_bound = 1;
      }
      else {
        window_bound = shape_in.At(i);
      }

      slice_sizes.push_back(window_bound);

      if (i < axis) {
        dim_numbers.add_offset_dims(i);
      }
      else if (i >= (axis + num_index_dims)) {
        int64_t indices_rank = shape_indices.NumAxes() - 1;
        dim_numbers.add_offset_dims(i + indices_rank - num_index_dims);
      }
    }
    
    dim_numbers.set_index_vector_dim(shape_indices.NumAxes() - 1);
    for (int64_t i = axis; i < axis + num_index_dims; i++) {
      dim_numbers.add_start_index_map(i);
    }
    ctx->SetOutput("out", xla::Gather(in, indices, dim_numbers, slice_sizes));
  }
};

REGISTER_XLA_OP_COMPILER(BatchGather, BatchGatherOp);

}  // namespace mola
}  // namespace oneflow
