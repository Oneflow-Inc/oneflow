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
#include "oneflow/core/common/memory_format.pb.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/stride.h"

namespace oneflow {

inline MemoryFormat GetMemoryFormatFromString(const std::string& memory_format_str) {
  if (memory_format_str == "contiguous") {
    return kContiguous;
  } else if (memory_format_str == "channels_last") {
    return kChannelsLast;
  }
  THROW(InvalidValueError) << "GetMemoryFormatFromString: Invalid memory_format_str";
}

inline std::string GetStringFromMemoryFormat(MemoryFormat memory_format) {
  if (memory_format == kContiguous) {
    return "contiguous";
  } else if (memory_format == kChannelsLast) {
    return "channels_last";
  } else if (memory_format == kPreserve) {
    return "preserve";
  }
  THROW(InvalidValueError) << "GetStringFromMemoryFormat: Invalid memory_format";
}

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

inline Shape GetChannelsLastShape2d(const ShapeView& shape) {
  int64_t ndim = shape.NumAxes();
  CHECK_OR_THROW(ndim == 3 || ndim == 4) << "GetChannelsLastShape2d doesn't support size " << ndim;

  Shape channels_last_2d_shape(ndim);
  channels_last_2d_shape[0] = shape[0];
  channels_last_2d_shape[ndim - 1] = shape[1];
  for (size_t i = 1; i < ndim - 1; i++) { channels_last_2d_shape[i] = shape[i + 1]; }
  return channels_last_2d_shape;
}

inline Shape GetShapeFromMemoryFormat(const ShapeView& shape, MemoryFormat memory_format) {
  if (memory_format == kContiguous) {
    return Shape(shape);
  } else if (memory_format == kChannelsLast) {
    return GetChannelsLastShape2d(shape);
  }
  THROW(InvalidValueError) << "GetStringFromMemoryFormat: Invalid memory_format";
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