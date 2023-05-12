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
#ifndef ONEFLOW_COMMON_MEMORY_FORMAT_UTIL_H_
#define ONEFLOW_COMMON_MEMORY_FORMAT_UTIL_H_

#include "oneflow/core/common/memory_format.pb.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/stride.h"

namespace oneflow {

// conversion between MemoryFormat and std::string
MemoryFormat GetMemoryFormatFromString(const std::string& memory_format_str);
std::string GetStringFromMemoryFormat(MemoryFormat memory_format);

//
MemoryFormat InferMemoryFormat(const ShapeView& shape_view, const Stride& stride);

Stride GetChannelsLastStrides2d(const Stride& stride);
Stride GetChannelsLastStrides2d(const ShapeView& shape, const Stride& stride);
Stride GetStrideFromMemoryFormat(const Stride& stride, MemoryFormat memory_format);

Stride GetStrideFromMemoryFormat(const ShapeView& shape, const Stride& stride,
                                 MemoryFormat memory_format);
Stride GetChannelsLastStrides2d(const ShapeView& shape);
Shape GetChannelsLastShape2d(const ShapeView& shape);
Shape GetShapeFromMemoryFormat(const ShapeView& shape, MemoryFormat memory_format);
bool IsContiguousInChannalsLast2d(const ShapeView& shape_view, const Stride& stride);

bool IsContiguous(const Shape& shape, const Stride& stride, MemoryFormat memory_format);
bool IsContiguous(const Shape& shape, const Stride& stride);

}  // namespace oneflow

#endif  // ONEFLOW_COMMON_MEMORY_FORMAT_UTIL_H_