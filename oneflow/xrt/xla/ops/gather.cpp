/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
    Shape input_shape = ctx->InputShape("in_0");
    Shape indices_shape = ctx->InputShape("indices_0");
    CHECK_GT(input_shape.NumAxes(), 0);
    CHECK_GT(indices_shape.NumAxes(), 0);
    CHECK_LE(indices_shape.NumAxes(), input_shape.NumAxes());

    xla::XlaOp input = ctx->Input("in_0");
    xla::XlaOp indices = ctx->Input("indices_0");
    int axis = GatherAxis(ctx);
    int batch_dims = GatherBatchDims(ctx);

    xla::XlaOp output;
    if (batch_dims > 0) {
      output = xla::TorchIndexSelect(input, indices, axis, batch_dims);
    } else {
      output = GenericGather(input, indices, input_shape, indices_shape, axis);
    }
    ctx->SetOutput("out_0", output);
  }

  virtual int GatherAxis(XlaOpContext *ctx) const { return ctx->Attr<int64_t>("axis"); }
  virtual int GatherBatchDims(XlaOpContext *ctx) const { return 0; }
};

class BatchGatherOp : public GatherOp {
 public:
  int GatherAxis(XlaOpContext *ctx) const override {
    Shape indices_shape = ctx->InputShape("indices_0");
    return indices_shape.NumAxes() - 1;
  }
  int GatherBatchDims(XlaOpContext *ctx) const override {
    Shape indices_shape = ctx->InputShape("indices_0");
    return indices_shape.NumAxes() - 1;
  }
};

REGISTER_XLA_OP_KERNEL(Gather, GatherOp).Finalize();
REGISTER_XLA_OP_KERNEL(BatchGather, BatchGatherOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
