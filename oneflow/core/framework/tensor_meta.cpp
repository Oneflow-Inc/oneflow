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
#include "oneflow/core/framework/device.h"

namespace oneflow {
namespace one {

MirroredTensorMeta::MirroredTensorMeta()
    : TensorMeta(std::make_shared<const Shape>(), std::make_shared<const Stride>(),
                 DataType::kInvalidDataType),
      device_(Symbol<Device>()),
      storage_offset_(0) {}

MirroredTensorMeta::MirroredTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                                       Symbol<Device> device)
    : TensorMeta(shape, std::make_shared<const Stride>(*shape), dtype),
      device_(device),
      storage_offset_(0) {}

MirroredTensorMeta::MirroredTensorMeta(const std::shared_ptr<const Shape>& shape,
                                       const std::shared_ptr<const Stride>& stride, DataType dtype,
                                       Symbol<Device> device, int64_t storage_offset)
    : TensorMeta(shape, stride, dtype), device_(device), storage_offset_(storage_offset) {}

bool MirroredTensorMeta::operator==(const MirroredTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && *this->device() == *other.device() && this->stride() == other.stride()
         && this->storage_offset() == other.storage_offset();
}

size_t MirroredTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Device>()(*device()) ^ std::hash<Stride>()(stride()) ^ storage_offset();
}

bool ConsistentTensorMeta::operator==(const ConsistentTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->nd_sbp() == other.nd_sbp() && this->parallel_desc() == other.parallel_desc();
}

size_t ConsistentTensorMeta::CalcHashValue() const {
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Symbol<NdSbp>>()(nd_sbp())
         ^ std::hash<Symbol<ParallelDesc>>()(parallel_desc());
}

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
      if (stride.at(i) != expected_stride) { contig_if_nonempty = false; }
      expected_stride *= shape.At(i);
    }
  }
  return contig_if_nonempty;
}

}  // namespace one
}  // namespace oneflow
