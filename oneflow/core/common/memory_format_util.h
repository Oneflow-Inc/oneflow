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
#pragma once
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/stride.h"

namespace oneflow {

inline Stride GetChannelsLastStrides2d(const ShapeView& shape, Stride& stride) {
  switch (shape.size()) {
    case 4:
      stride[1] = 1;
      stride[3] = shape[1];
      stride[2] = stride[3] * shape[3];
      stride[0] = stride[2] * shape[2];
      return stride;
    case 3:
      stride[0] = 1;
      stride[2] = shape[0];
      stride[1] = stride[2] * shape[2];
      return stride;
    default: CHECK_OR_THROW(false) << "ChannelsLast2d doesn't support size " << shape.size();
  }
}

inline Stride GetChannelsLastStrides2d(const ShapeView& shape) {
  Stride stride(shape.size());
  GetChannelsLastStrides2d(shape, stride);
  return stride;
}

inline bool IsContiguousInChannalsLast2d(const ShapeView& shape_view, const Stride& stride) {
  if (shape_view.size() < 4) { return false; }
  int64_t total_size = 1;
  for (auto& d : {1, 3, 2, 0}) {
    int64_t size_d = shape_view.At(d);
    if (size_d != 1) {
      if (stride.at(d) != total_size) { return false; }
      total_size *= size_d;
    }
  }
  return true;
}

}  // namespace oneflow