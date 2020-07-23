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
#ifndef ONEFLOW_XRT_XLA_XLA_SHAPE_H_
#define ONEFLOW_XRT_XLA_XLA_SHAPE_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace oneflow {
namespace xrt {
namespace mola {

Shape XlaShapeToOfShape(const xla::Shape &xla_shape);

xla::Shape OfShapeToXlaShape(const Shape &shape, DataType dtype);

xla::Shape OfShapeToXlaShape(const Shape &shape, xla::PrimitiveType type);

// Returns shape[start_dim:end_dim]
Shape SliceShape(const Shape &shape, size_t start_dim, size_t end_dim);

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_SHAPE_H_
