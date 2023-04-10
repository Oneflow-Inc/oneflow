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
#include "oneflow/cambricon/kernels/convert_memory_format_util.h"

#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace mlu {

std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(DeviceType::kMLU, num_dims);
}

void ComputeIdentity(ep::Stream* stream, int ndim, const int64_t* shape, DataType data_type,
                     const void* in, void* out) {
  size_t count = 1;
  for (int i = 0; i < ndim; ++i) { count *= shape[i]; }
  Memcpy<DeviceType::kMLU>(stream, out, in, count * GetSizeOfDataType(data_type));
}

void ComputeNchwToNhwc(ep::Stream* stream, int ndim, const int64_t* shape, DataType data_type,
                       const void* in, void* out) {
  if (ndim <= 2) { return ComputeIdentity(stream, ndim, shape, data_type, in, out); }

  std::vector<int32_t> permute(ndim);
  permute[0] = 0;
  permute[ndim - 1] = 1;
  for (int i = 0; i < ndim - 2; ++i) { permute[i + 1] = i + 2; }
  auto primitive = NewPermutePrimitive(ndim);
  CHECK_NOTNULL_OR_THROW(primitive);
  primitive->Launch(stream, data_type, ndim, shape, in, permute.data(), out);
}

void ComputeNhwcToNchw(ep::Stream* stream, int ndim, const int64_t* shape, DataType data_type,
                       const void* in, void* out) {
  if (ndim <= 2) { return ComputeIdentity(stream, ndim, shape, data_type, in, out); }

  std::vector<int32_t> permute(ndim);
  permute[0] = 0;
  permute[1] = ndim - 1;
  for (int i = 0; i < ndim - 2; ++i) { permute[i + 2] = i + 1; }
  auto primitive = NewPermutePrimitive(ndim);
  CHECK_NOTNULL_OR_THROW(primitive);
  primitive->Launch(stream, data_type, ndim, shape, in, permute.data(), out);
}

using ConvertMemoryFormatFunc =
    std::function<void(ep::Stream*, int, const int64_t*, DataType, const void*, void*)>;

ConvertMemoryFormatFunc convert_funcs[MemoryFormat_Max][MemoryFormat_Max] = {
    /*kDefaukt->other*/ {ComputeIdentity, ComputeIdentity, ComputeNchwToNhwc},
    /*kNCHW->other*/ {ComputeIdentity, ComputeIdentity, ComputeNchwToNhwc},
    /*kNHWC->other*/ {ComputeNhwcToNchw, ComputeNhwcToNchw, ComputeIdentity},
};

void ConvertMemoryFormat(ep::Stream* stream, const user_op::Tensor* in, user_op::Tensor* out,
                         MemoryFormat in_memory_format, MemoryFormat out_memory_format) {
  auto convert_func = convert_funcs[in_memory_format][out_memory_format];
  convert_func(stream, in->shape_view().size(), in->shape_view().data(), in->data_type(),
               in->dptr(), out->mut_dptr());
}

void ConvertMemoryFormat(ep::Stream* stream, int ndim, const int64_t* shape, DataType data_type,
                         const void* in, void* out, MemoryFormat in_memory_format,
                         MemoryFormat out_memory_format) {
  auto convert_func = convert_funcs[in_memory_format][out_memory_format];
  convert_func(stream, ndim, shape, data_type, in, out);
}

void ConvertMemoryFormat(ep::Stream* stream, const ShapeView& shape, DataType data_type,
                         const void* in, void* out, MemoryFormat in_memory_format,
                         MemoryFormat out_memory_format) {
  ConvertMemoryFormat(stream, shape.size(), shape.data(), data_type, in, out, in_memory_format,
                      out_memory_format);
}

}  // namespace mlu
}  // namespace oneflow
