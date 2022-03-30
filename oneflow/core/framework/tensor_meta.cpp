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
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/common/stride.h"

namespace oneflow {
namespace one {

bool IsContiguous(const Shape& shape, const Stride& stride) {
  if (!shape.is_initialized() || shape.NumAxes() < 1 || shape.elem_cnt() <= 1) { return true; }
  int64_t dim = shape.NumAxes();
  int64_t expected_stride = 1;
  bool contig_if_nonempty = true;
  for (int64_t i = dim - 1; i >= 0; --i) {
    // Contiguous by default when any dim is equal to zero
    // https://stackoverflow.com/questions/31681324/identify-contiguous-segments-of-a-non-contiguous-numpy-array
    if (shape.At(i) == 0) { return true; }
    if (contig_if_nonempty && shape.At(i) != 1) {
      if (stride.At(i) != expected_stride) { contig_if_nonempty = false; }
      expected_stride *= shape.At(i);
    }
  }
  return contig_if_nonempty;
}

bool IsContiguous(const DimVector& shape_vec, const StrideVector& stride_vec) {
  const size_t ndim = shape_vec.size();
  const size_t stride_ndim = stride_vec.size();
  if (ndim < 1 || stride_ndim < 1) { return true; }
  int64_t elem_cnt = 1;
  for (int64_t s : shape_vec) { elem_cnt *= s; }
  if (elem_cnt <= 1) { return true; }

  int64_t expected_stride = 1;
  bool contig_if_nonempty = true;
  for (int64_t i = ndim - 1; i >= 0; --i) {
    if (shape_vec.at(i) == 0) { return true; }
    if (contig_if_nonempty && shape_vec.at(i) != 1) {
      if (stride_vec.at(i) != expected_stride) { contig_if_nonempty = false; }
      expected_stride *= shape_vec.at(i);
    }
  }
  return contig_if_nonempty;
}

bool IsContiguous(const user_op::Tensor* tensor) {
  const StrideVector& stride_vec = tensor->stride().StrideVec();
  DimVector shape_vec;
  tensor->shape().ToDimVector(&shape_vec);
  return IsContiguous(shape_vec, stride_vec);
}

StrideParam get_StrideParam(const user_op::Tensor* tensor) {
  const int32_t ndim = tensor->shape().NumAxes();
  const StrideVector& stride_vec = tensor->stride().StrideVec();
  DimVector shape_vec;
  tensor->shape().ToDimVector(&shape_vec);
  return StrideParam(stride_vec.data(), ndim);
}

}  // namespace one
}  // namespace oneflow
