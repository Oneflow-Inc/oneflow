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
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace mlu {

inline Shape ComputeShapeNchwToNhwc(const Shape& shape) {
  int ndim = shape.size();
  if (ndim <= 2) { return shape; }
  Shape target_shape(ndim);
  target_shape[0] = shape[0];
  target_shape[ndim - 1] = shape[1];
  for (int i = 0; i < ndim - 2; ++i) { target_shape[i + 1] = shape[i + 2]; }
  return target_shape;
}

inline Shape ComputeShapeNhwcToNchw(const Shape& shape) {
  int ndim = shape.size();
  if (ndim <= 2) { return shape; }
  Shape target_shape(ndim);
  target_shape[0] = shape[0];
  target_shape[1] = shape[ndim - 1];
  for (int i = 0; i < ndim - 2; ++i) { target_shape[i + 2] = shape[i + 1]; }
  return target_shape;
}

void ConvertMemoryFormat(ep::Stream* stream, const user_op::Tensor* in, user_op::Tensor* out,
                         MemoryFormat in_memory_format, MemoryFormat out_memory_format);

void ConvertMemoryFormat(ep::Stream* stream, int ndim, const int64_t* shape, DataType data_type,
                         const void* in, void* out, MemoryFormat in_memory_format,
                         MemoryFormat out_memory_format);

void ConvertMemoryFormat(ep::Stream* stream, const ShapeView& shape, DataType data_type,
                         const void* in, void* out, MemoryFormat in_memory_format,
                         MemoryFormat out_memory_format);

}  // namespace mlu
}  // namespace oneflow
