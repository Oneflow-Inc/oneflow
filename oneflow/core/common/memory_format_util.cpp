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
#include "oneflow/core/common/memory_format.pb.h"
#include "oneflow/core/common/memory_format_util.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/throw.h"

namespace oneflow {

MemoryFormat GetMemoryFormatFromString(const std::string& memory_format_str) {
  if (memory_format_str == "contiguous" || memory_format_str == "channels_first") {
    return kContiguous;
  } else if (memory_format_str == "channels_last") {
    return kChannelsLast;
  }
  THROW(InvalidValueError) << "GetMemoryFormatFromString: Invalid memory_format_str";
}

std::string GetStringFromMemoryFormat(MemoryFormat memory_format) {
  if (memory_format == kContiguous) {
    return "channels_first";
  } else if (memory_format == kChannelsLast) {
    return "channels_last";
  } else if (memory_format == kPreserve) {
    return "preserve";
  }
  THROW(InvalidValueError) << "GetStringFromMemoryFormat: Invalid memory_format";
}

bool IsContiguous(const ShapeView& shape_view, const Stride& stride) {
  if (shape_view.NumAxes() < 1 || shape_view.elem_cnt() <= 1) { return true; }
  int64_t dim = shape_view.NumAxes();
  int64_t expected_stride = 1;
  bool contig_if_nonempty = true;
  for (int64_t i = dim - 1; i >= 0; --i) {
    // Contiguous by default when any dim is equal to zero
    // https://stackoverflow.com/questions/31681324/identify-contiguous-segments-of-a-non-contiguous-numpy-array
    if (shape_view.At(i) == 0) { return true; }
    if (contig_if_nonempty && shape_view.At(i) != 1) {
      if (stride.at(i) != expected_stride) { contig_if_nonempty = false; }
      expected_stride *= shape_view.At(i);
    }
  }
  return contig_if_nonempty;
}

bool IsContiguous(const Shape& shape, const Stride& stride, MemoryFormat memory_format) {
  if (!shape.is_initialized()) { return true; }
  if (memory_format == MemoryFormat::kContiguous) {
    return IsContiguous(ShapeView(shape), stride);
  } else if (memory_format == MemoryFormat::kChannelsLast) {
    return IsContiguousInChannalsLast2d(shape, stride);
  }
  CHECK_OR_THROW(false) << "Unimplemented memory_format: " << memory_format;
}

bool IsContiguousInChannalsLast2d(const ShapeView& shape_view, const Stride& stride) {
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

MemoryFormat InferMemoryFormat(const ShapeView& shape_view, const Stride& stride) {
  if (IsContiguous(shape_view, stride)) {
    return kContiguous;
  } else if (IsContiguousInChannalsLast2d(shape_view, stride)) {
    return kChannelsLast;
  }
  THROW(InvalidValueError)
      << "InferMemoryFormat: cannot infer memory format according to shape and stride";
}

Stride GetChannelsLastStrides2d(const Stride& stride) {
  Stride result(stride.size());
  // 4d: [HWC, HW, W, 1] -> [HWC, 1, WC, C]
  switch (stride.size()) {
    case 4:
      result[1] = 1;
      result[0] = stride[0];
      result[3] = stride[0] / stride[1];
      result[2] = result[3] * stride[2];
    default:
      CHECK_OR_THROW(false)
          << "GetChannelsLastStrides2d: channels_last_2d memory format doesn't support size "
          << stride.size();
  }
  return stride;
}

Stride GetChannelsLastStrides2d(const ShapeView& shape, const Stride& stride) {
  Stride result(stride.size());
  switch (shape.size()) {
    case 4:
      result[1] = 1;
      result[3] = shape[1];
      result[2] = stride[3] * shape[3];
      result[0] = stride[2] * shape[2];
    case 3:
      result[0] = 1;
      result[2] = shape[0];
      result[1] = stride[2] * shape[2];
    default:
      CHECK_OR_THROW(false)
          << "GetChannelsLastStrides2d: channels_last_2d memory format doesn't support size "
          << shape.size();
      return result;
  }
}

Stride GetStrideFromMemoryFormat(const Stride& stride, MemoryFormat memory_format) {
  if (memory_format == kContiguous) {
    return stride;
  } else if (memory_format == kChannelsLast) {
    return GetChannelsLastStrides2d(stride);
  } else {
    CHECK_OR_THROW(false) << "GetStrideFromMemoryFormat: invalid memory_format";
  }
}

Stride GetStrideFromMemoryFormat(const ShapeView& shape, const Stride& stride,
                                 MemoryFormat memory_format) {
  if (memory_format == kContiguous) {
    return stride;
  } else if (memory_format == kChannelsLast) {
    return GetChannelsLastStrides2d(shape, stride);
  } else {
    CHECK_OR_THROW(false) << "GetStrideFromMemoryFormat: invalid memory_format";
  }
}

Stride GetChannelsLastStrides2d(const ShapeView& shape) {
  Stride stride(shape.size());
  GetChannelsLastStrides2d(shape, stride);
  return stride;
}

Shape GetChannelsLastShape2d(const ShapeView& shape) {
  int64_t ndim = shape.NumAxes();
  CHECK_OR_THROW(ndim == 3 || ndim == 4)
      << "GetChannelsLastShape2d: channels_last_2d memory format doesn't support size " << ndim;
  Shape channels_last_2d_shape(ndim);
  channels_last_2d_shape[0] = shape[0];
  channels_last_2d_shape[ndim - 1] = shape[1];
  for (size_t i = 1; i < ndim - 1; i++) { channels_last_2d_shape[i] = shape[i + 1]; }
  return channels_last_2d_shape;
}

Shape GetShapeFromMemoryFormat(const ShapeView& shape, MemoryFormat memory_format) {
  if (memory_format == kContiguous) {
    return Shape(shape);
  } else if (memory_format == kChannelsLast) {
    return GetChannelsLastShape2d(shape);
  }
  THROW(InvalidValueError) << "GetStringFromMemoryFormat: Invalid memory_format";
}

// bool IsContiguous(const Shape& shape, const Stride& stride, MemoryFormat memory_format) {
//   if (!shape.is_initialized()) { return true; }
//   if (memory_format == MemoryFormat::kContiguous) {
//     return IsContiguous(ShapeView(shape), stride);
//   } else if (memory_format == MemoryFormat::kChannelsLast) {
//     return IsContiguousInChannalsLast2d(shape, stride);
//   }
//   CHECK_OR_THROW(false) << "Unimplemented memory_format: " << memory_format;
// }

// bool IsContiguous(const ShapeView& shape_view, const Stride& stride) {
//   if (shape_view.NumAxes() < 1 || shape_view.elem_cnt() <= 1) { return true; }
//   int64_t dim = shape_view.NumAxes();
//   int64_t expected_stride = 1;
//   bool contig_if_nonempty = true;
//   for (int64_t i = dim - 1; i >= 0; --i) {
//     // Contiguous by default when any dim is equal to zero
//     //
//     https://stackoverflow.com/questions/31681324/identify-contiguous-segments-of-a-non-contiguous-numpy-array
//     if (shape_view.At(i) == 0) { return true; }
//     if (contig_if_nonempty && shape_view.At(i) != 1) {
//       if (stride.at(i) != expected_stride) { contig_if_nonempty = false; }
//       expected_stride *= shape_view.At(i);
//     }
//   }
//   return contig_if_nonempty;
// }

}  // namespace oneflow