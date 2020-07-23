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
#include "oneflow/xrt/api.h"
#include "oneflow/xrt/xla/xla_shape.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/xrt/xla/xla_data_type.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include <vector>

namespace oneflow {
namespace xrt {
namespace mola {

Shape XlaShapeToOfShape(const xla::Shape &xla_shape) {
  CHECK(!xla_shape.IsTuple());
  int rank = xla_shape.rank();
  std::vector<int64_t> dimensions(rank);
  for (int i = 0; i < rank; ++i) { dimensions[i] = xla_shape.dimensions(i); }
  return AsShape(dimensions);
}

xla::Shape OfShapeToXlaShape(const Shape &shape, DataType dtype) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(dtype);
  return OfShapeToXlaShape(shape, type);
}

xla::Shape OfShapeToXlaShape(const Shape &shape, xla::PrimitiveType type) {
  int rank = shape.NumAxes();
  std::vector<long long> layout(rank);
  std::vector<long long> dimensions(rank);
  for (int i = 0; i < rank; ++i) { dimensions[i] = shape.At(i); }

  std::iota(layout.rbegin(), layout.rend(), 0);
  return xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
}

Shape SliceShape(const Shape &shape, size_t start_dim, size_t end_dim) {
  CHECK_LE(start_dim, end_dim);
  CHECK_LE(end_dim, shape.NumAxes());

  std::vector<int64_t> slice_shape(end_dim - start_dim);
  for (size_t i = start_dim; i < end_dim; ++i) { slice_shape[i] = shape.At(i); }
  return AsShape(slice_shape);
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
