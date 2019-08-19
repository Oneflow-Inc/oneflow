#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "oneflow/xla/of2xla/xla_helpers.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "absl/types/span.h"
namespace oneflow {
namespace mola {

xla::XlaOp One(xla::XlaBuilder *builder, DataType data_type) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return xla::ConstantLiteral(builder, xla::LiteralUtil::One(type));
}

xla::XlaOp Zero(xla::XlaBuilder *builder, DataType data_type) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return xla::ConstantLiteral(builder, xla::LiteralUtil::Zero(type));
}

xla::XlaOp IntegerLiteral(xla::XlaBuilder *builder, DataType data_type,
                          int32_t value) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return ::tensorflow::IntegerLiteral(builder, type, value);
}

xla::XlaOp FloatLiteral(xla::XlaBuilder *builder, DataType data_type,
                        float value) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return ::tensorflow::FloatLiteral(builder, type, value);
}

xla::XlaOp Reshape(xla::XlaOp input, Shape dest_shape) {
  std::vector<long long> shape_dim(dest_shape.NumAxes());
  for (int i = 0; i < dest_shape.NumAxes(); ++i) {
    shape_dim[i] = dest_shape.At(i);
  }
  return xla::Reshape(input, shape_dim);
}

xla::XlaOp XlaScatter(
    const xla::XlaOp& buffer, const xla::XlaOp& updates,
    const xla::XlaOp& indices, bool indices_are_vectors,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp, xla::XlaBuilder*)>&
        combiner,
    xla::XlaBuilder* builder) {
  OF_CHECK_AND_ASSIGN(xla::Shape buffer_shape, builder->GetShape(buffer));
  OF_CHECK_AND_ASSIGN(xla::Shape updates_shape, builder->GetShape(updates));
  OF_CHECK_AND_ASSIGN(xla::Shape indices_shape, builder->GetShape(indices));
  absl::Span<const long long> indices_dims =
      xla::AsInt64Slice(indices_shape.dimensions());

  // If the indices are N-dimensional, the minor dimension of indices contains
  // the indices to update. Otherwise the indices are all scalars.
  int64_t num_index_dims = 1;
  if (indices_are_vectors) {
    CHECK_EQ(indices_dims.empty(), 0);
    num_index_dims = indices_dims.back();
    CHECK_LE(num_index_dims, buffer_shape.rank());
    indices_dims.remove_suffix(1);
  }

  int64_t num_indices = 1;
  for (int64_t dim : indices_dims) {
    num_indices *= dim;
  }

  // Degenerate case: nothing to update. Return the buffer unchanged.
  if (num_indices == 0) {
    return buffer;
  }

  // If any of the indexed dimensions are zero in the buffer, the update cannot
  // succeed since it updates a slice of size 1.
  for (int64_t i = 0; i < num_index_dims; ++i) {
    CHECK_EQ(xla::ShapeUtil::GetDimension(buffer_shape, i), 0);
  }

  xla::ScatterDimensionNumbers dim_numbers;
  dim_numbers.set_index_vector_dim(indices_are_vectors
                                       ? indices_shape.dimensions_size() - 1
                                       : indices_shape.dimensions_size());

  int64_t updates_rank = updates_shape.rank();
  int64_t buffer_rank = buffer_shape.rank();
  int64_t num_window_dims_in_updates = buffer_rank - num_index_dims;

  // If the rank of `updates` is 0 and does not match the expected rank of
  // updates, broadcast `updates` to the expected shape of updates.
  auto new_updates = updates;
  std::vector<long long> expected_updates_dims(indices_dims.begin(),
                                           indices_dims.end());
  for (int64_t dim = num_index_dims; dim < buffer_rank; ++dim) {
    expected_updates_dims.push_back(buffer_shape.dimensions(dim));
  }
  int64_t expected_updates_rank = expected_updates_dims.size();
  if (updates_rank == 0 && expected_updates_rank != 0) {
    new_updates = xla::Broadcast(updates, expected_updates_dims);
    OF_CHECK_AND_ASSIGN(updates_shape, builder->GetShape(new_updates));
    updates_rank = updates_shape.rank();
  }

  if (updates_rank > 0) {
    for (int64_t i = (updates_rank - num_window_dims_in_updates);
         i < updates_rank; ++i) {
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
    auto xla_scalar_shape =
        xla::ShapeUtil::MakeShape(buffer_shape.element_type(), {});
    auto p0 = xla::Parameter(&cb, 0, xla_scalar_shape, "p0");
    auto p1 = xla::Parameter(&cb, 1, xla_scalar_shape, "p1");
    if (combiner) {
      combiner(p0, p1, &cb);
    }
    combiner_computation = cb.Build().ConsumeValueOrDie();
  }
  return xla::Scatter(buffer, indices, new_updates, combiner_computation,
                      dim_numbers);
}


}  // namespace mola
}  // namespace oneflow
