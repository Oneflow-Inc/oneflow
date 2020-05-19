#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/api.h"
#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

xla::XlaOp GenericGather(const xla::XlaOp &input, const xla::XlaOp &indices,
                         const Shape &input_shape, const Shape &indices_shape, int64_t axis) {
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
xla::XlaOp GenericGatherGrad(
    const xla::XlaOp &buffer, const xla::XlaOp &updates, const xla::XlaOp &indices,
    bool indices_are_vectors,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp, xla::XlaBuilder *)> &combiner,
    xla::XlaBuilder *builder) {
  MOLA_CHECK_AND_ASSIGN(xla::Shape buffer_shape, builder->GetShape(buffer));
  MOLA_CHECK_AND_ASSIGN(xla::Shape updates_shape, builder->GetShape(updates));
  MOLA_CHECK_AND_ASSIGN(xla::Shape indices_shape, builder->GetShape(indices));
  absl::Span<const long long> indices_dims = xla::AsInt64Slice(indices_shape.dimensions());

  // If the indices are N-dimensional, the minor dimension of indices contains
  // the indices to update. Otherwise the indices are all scalars.
  int64_t num_index_dims = 1;
  if (indices_are_vectors) {
    num_index_dims = indices_dims.back();
    indices_dims.remove_suffix(1);
  }

  int64_t num_indices = 1;
  for (int64_t dim : indices_dims) { num_indices *= dim; }

  // Degenerate case: nothing to update. Return the buffer unchanged.
  if (num_indices == 0) { return buffer; }

  xla::ScatterDimensionNumbers dim_numbers;
  dim_numbers.set_index_vector_dim(indices_are_vectors ? indices_shape.dimensions_size() - 1
                                                       : indices_shape.dimensions_size());

  int64_t updates_rank = updates_shape.rank();
  int64_t buffer_rank = buffer_shape.rank();
  int64_t num_window_dims_in_updates = buffer_rank - num_index_dims;

  // If the rank of `updates` is 0 and does not match the expected rank of
  // updates, broadcast `updates` to the expected shape of updates.
  auto new_updates = updates;
  std::vector<long long> expected_updates_dims(indices_dims.begin(), indices_dims.end());
  for (int64_t dim = num_index_dims; dim < buffer_rank; ++dim) {
    expected_updates_dims.push_back(buffer_shape.dimensions(dim));
  }
  int64_t expected_updates_rank = expected_updates_dims.size();
  if (updates_rank == 0 && expected_updates_rank != 0) {
    new_updates = xla::Broadcast(updates, expected_updates_dims);
    MOLA_CHECK_AND_ASSIGN(updates_shape, builder->GetShape(new_updates));
    updates_rank = updates_shape.rank();
  }
  if (updates_rank > 0) {
    for (int64_t i = (updates_rank - num_window_dims_in_updates); i < updates_rank; ++i) {
      dim_numbers.add_update_window_dims(i);
    }
  }

  for (int64_t i = 0; i < num_index_dims; ++i) {
    dim_numbers.add_inserted_window_dims(i);
    dim_numbers.add_scatter_dims_to_operand_dims(i);
  }

  // Build the combiner computation.
  xla::XlaComputation combiner_computation;
  {
    xla::XlaBuilder cb("scatter-combiner");
    auto xla_scalar_shape = xla::ShapeUtil::MakeShape(buffer_shape.element_type(), {});
    auto p0 = xla::Parameter(&cb, 0, xla_scalar_shape, "p0");
    auto p1 = xla::Parameter(&cb, 1, xla_scalar_shape, "p1");
    if (combiner) { combiner(p0, p1, &cb); }
    combiner_computation = cb.Build().ConsumeValueOrDie();
  }
  return xla::Scatter(buffer, indices, new_updates, combiner_computation, dim_numbers);
}

class GatherOp : public XlaOpKernel {
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
      output = GenericGather(input, indices, input_shape, indices_shape, axis);
    }
    ctx->SetOutput("out", output);
  }

  virtual int GatherAxis(XlaOpContext *ctx) const { return ctx->Attr<int64_t>("axis"); }
  virtual int GatherBatchDims(XlaOpContext *ctx) const { return 0; }
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

REGISTER_XLA_OP_KERNEL(Gather, GatherOp).Finalize();
REGISTER_XLA_OP_KERNEL(BatchGather, BatchGatherOp).Finalize();

class GatherGradOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaBuilder *builder = ctx->builder();
    xla::XlaOp updates = ctx->Input("out_diff");
    xla::XlaOp indices = ctx->Input("indices");
    int64_t gather_dim_size = ctx->Attr<int64_t>("gather_dim_size");
    int64_t axis = ctx->Attr<int64_t>("axis");

    Shape updates_shape = ctx->InputShape("out_diff");
    Shape indices_shape = ctx->InputShape("indices");
    DataType updates_data_type = ctx->InputType("out_diff");

    const auto &indices_dim_vec = indices_shape.dim_vec();
    const auto &updates_dim_vec = updates_shape.dim_vec();
    CHECK_LT(axis, updates_dim_vec.size());
    std::vector<int64_t> buffer_dim_vec;

    for (int i = 0; i < axis; ++i) { buffer_dim_vec.push_back(updates_dim_vec[i]); }
    buffer_dim_vec.push_back(gather_dim_size);
    for (int i = axis + indices_dim_vec.size(); i < updates_dim_vec.size(); ++i) {
      buffer_dim_vec.push_back(updates_dim_vec[i]);
    }

    xla::XlaOp buffer = Zeros(ctx->builder(), AsShape(buffer_dim_vec), updates_data_type);
    auto combiner = [](xla::XlaOp x, xla::XlaOp y, xla::XlaBuilder *) { return xla::Add(x, y); };
    ctx->SetOutput("in_diff", GenericGatherGrad(buffer, updates, indices, true, combiner, builder));
  }
};

REGISTER_XLA_OP_KERNEL(GatherGrad, GatherGradOp).Finalize();

class UnsortedSegmentSumOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp segment_ids = ctx->Input("segment_ids");
    xla::XlaOp data = ctx->Input("segment_ids");
    Shape data_shape = ctx->InputShape("data");
    Shape segment_ids_shape = ctx->InputShape("segment_ids");
    DataType data_type = ctx->InputType("data");
    int64_t num_segments = ctx->Attr<int64_t>("num_segments");
    std::vector<int64_t> buffer_dim_vec = InitBufferDimVec(ctx);

    buffer_dim_vec.push_back(num_segments);
    for (int i = Axis(ctx) + segment_ids_shape.NumAxes(); i < data_shape.dim_vec().size(); ++i) {
      buffer_dim_vec.push_back(data_shape.dim_vec()[i]);
    }

    const int64_t num_elems = segment_ids_shape.NumAxes() > 0 ? segment_ids_shape.At(0) : 1;

    if (data_shape.NumAxes() == 0 && num_elems != 1) { data = xla::Broadcast(data, {num_elems}); }

    xla::XlaOp default_value = Zeros(ctx->builder(), AsShape(buffer_dim_vec), data_type);
    xla::XlaBuilder *builder = ctx->builder();
    std::vector<long long> buffer_dim_vecs(buffer_dim_vec.size());

    xla::XlaOp buffer = xla::Broadcast(default_value, buffer_dim_vecs);
    auto combiner = [](xla::XlaOp x, xla::XlaOp y, xla::XlaBuilder *) { return xla::Add(x, y); };
    ctx->SetOutput("out", GenericGatherGrad(buffer, data, segment_ids, false, combiner, builder));
  }

  virtual int Axis(XlaOpContext *ctx) const { return ctx->Attr<int64_t>("axis"); }

  virtual std::vector<int64_t> InitBufferDimVec(XlaOpContext *ctx) const {
    std::vector<int64_t> buffer_dim_vec;
    const auto data_dim_vec = ctx->InputShape("data").dim_vec();
    for (int i = 0; i < ctx->Attr<int64_t>("axis"); ++i) {
      buffer_dim_vec.push_back(data_dim_vec[i]);
    }
    return std::move(buffer_dim_vec);
  }
};

class UnsortedBatchSegmentSumOp : public UnsortedSegmentSumOp {
 public:
  int Axis(XlaOpContext *ctx) const override { return 0; }

  std::vector<int64_t> InitBufferDimVec(XlaOpContext *ctx) const override {
    return {ctx->InputShape("segment_ids").At(0)};
  }
};

REGISTER_XLA_OP_KERNEL(UnsortedSegmentSum, UnsortedSegmentSumOp).Finalize();
REGISTER_XLA_OP_KERNEL(UnsortedBatchSegmentSum, UnsortedBatchSegmentSumOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
