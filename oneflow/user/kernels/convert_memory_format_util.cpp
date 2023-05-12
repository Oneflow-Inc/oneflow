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
#include "oneflow/user/kernels/convert_memory_format_util.h"

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/memory_format.pb.h"
#include "oneflow/core/common/memory_format_util.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(DeviceType device_type,
                                                            const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(device_type, num_dims);
}

std::unique_ptr<ep::primitive::Memcpy> NewMemcpyPrimitive(DeviceType device_type) {
  return ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(
      device_type, ep::primitive::MemcpyKind::kDtoD);
}

void ComputeIdentity(ep::Stream* stream, const ShapeView& shape, DataType data_type, const void* in,
                     void* out) {
  int64_t ndim = shape.NumAxes();
  size_t count = 1;
  for (int i = 0; i < ndim; ++i) { count *= shape[i]; }
  auto memcpy_primitive = NewMemcpyPrimitive(stream->device_type());
  CHECK(memcpy_primitive) << "Can not create Memcpy primitive for device type "
                          << stream->device_type();
  memcpy_primitive->Launch(stream, out, in, count * GetSizeOfDataType(data_type));
}

void ComputeContiguousToChannelsLast(ep::Stream* stream, const ShapeView& shape, DataType data_type,
                                     const void* in, void* out) {
  int64_t ndim = shape.NumAxes();
  if (ndim <= 2) { return ComputeIdentity(stream, shape, data_type, in, out); }

  // auto new_shape = GetShapeFromMemoryFormat(shape, kContiguous);

  std::vector<int32_t> permute(ndim);
  permute[0] = 0;
  permute[ndim - 1] = 1;
  for (int i = 0; i < ndim - 2; ++i) { permute[i + 1] = i + 2; }
  auto primitive = NewPermutePrimitive(stream->device_type(), ndim);
  CHECK_NOTNULL_OR_THROW(primitive);
  primitive->Launch(stream, data_type, ndim, shape.data(), in, permute.data(), out);
}

void ComputeChannelsLastToContiguous(ep::Stream* stream, const ShapeView& shape, DataType data_type,
                                     const void* in, void* out) {
  auto ndim = shape.NumAxes();
  if (ndim <= 2) { return ComputeIdentity(stream, shape, data_type, in, out); }

  auto shape_of_channels_last = GetShapeFromMemoryFormat(shape, kContiguous);
  std::vector<int32_t> permute(ndim);
  permute[0] = 0;
  permute[1] = ndim - 1;
  for (int i = 0; i < ndim - 2; ++i) { permute[i + 2] = i + 1; }
  auto primitive = NewPermutePrimitive(stream->device_type(), ndim);
  CHECK_NOTNULL_OR_THROW(primitive);
  primitive->Launch(stream, data_type, ndim, shape_of_channels_last.data(), in, permute.data(),
                    out);
}

using ConvertMemoryFormatFunc =
    std::function<void(ep::Stream*, const ShapeView&, DataType, const void*, void*)>;

void InvalidConvertFunc(ep::Stream*, const ShapeView&, DataType, const void*, void*) {
  CHECK_OR_THROW(false) << "ConvertMemoryFormat: Invalid memory format, please check the "
                           "memory_format of input and output.";
}

ConvertMemoryFormatFunc convert_funcs[kMemoryFormatCount][kMemoryFormatCount] = {
    /* kContiguous   -> others */ {ComputeIdentity, ComputeContiguousToChannelsLast,
                                   InvalidConvertFunc},
    /* kChannelsLast -> others */
    {ComputeChannelsLastToContiguous, ComputeIdentity, InvalidConvertFunc},
    /* kPreserve     -> others */ {InvalidConvertFunc, InvalidConvertFunc, InvalidConvertFunc},
};

void ConvertMemoryFormat(ep::Stream* stream, const user_op::Tensor* in, user_op::Tensor* out,
                         MemoryFormat in_memory_format, MemoryFormat out_memory_format) {
  auto convert_func = convert_funcs[in_memory_format][out_memory_format];
  convert_func(stream, in->shape_view(), in->data_type(), in->dptr(), out->mut_dptr());
}

// void ConvertMemoryFormat(ep::Stream* stream, int ndim, const int64_t* shape, DataType data_type,
//                          const void* in, void* out, MemoryFormat in_memory_format,
//                          MemoryFormat out_memory_format) {
//   auto convert_func = convert_funcs[in_memory_format][out_memory_format];
//   convert_func(stream, ndim, shape, data_type, in, out);
// }

void ConvertMemoryFormat(ep::Stream* stream, const ShapeView& shape, DataType data_type,
                         const void* in, void* out, MemoryFormat in_memory_format,
                         MemoryFormat out_memory_format) {
  auto convert_func = convert_funcs[in_memory_format][out_memory_format];
  convert_func(stream, shape, data_type, in, out);
}

}  // namespace oneflow
