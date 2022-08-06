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
#include "oneflow/core/common/tensor_meta.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {
namespace one {

MutTensorMeta::MutTensorMeta()
    : TensorMeta(std::make_shared<const Shape>(), std::make_shared<const Stride>(),
                 kInvalidDataType) {}

MutTensorMeta::MutTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype)
    : TensorMeta(shape, std::make_shared<const Stride>(*shape), dtype) {}

MutTensorMeta::MutTensorMeta(const std::shared_ptr<const Shape>& shape,
                             const std::shared_ptr<const Stride>& stride, DataType dtype)
    : TensorMeta(shape, stride, dtype) {}

bool MutTensorMeta::operator==(const MutTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->stride() == other.stride();
}

size_t MutTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Stride>()(stride());
}

LocalTensorMeta::LocalTensorMeta()
    : TensorMeta(std::make_shared<const Shape>(), std::make_shared<const Stride>(),
                 DataType::kInvalidDataType),
      device_(Symbol<Device>()),
      storage_offset_(0) {}

LocalTensorMeta::LocalTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                                 Symbol<Device> device)
    : TensorMeta(shape, std::make_shared<const Stride>(*shape), dtype),
      device_(device),
      storage_offset_(0) {}

LocalTensorMeta::LocalTensorMeta(const std::shared_ptr<const Shape>& shape,
                                 const std::shared_ptr<const Stride>& stride, DataType dtype,
                                 Symbol<Device> device, int64_t storage_offset)
    : TensorMeta(shape, stride, dtype), device_(device), storage_offset_(storage_offset) {}

bool LocalTensorMeta::operator==(const LocalTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->device() == other.device() && this->stride() == other.stride()
         && this->storage_offset() == other.storage_offset();
}

size_t LocalTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Symbol<Device>>()(device()) ^ std::hash<Stride>()(stride()) ^ storage_offset();
}

MutLocalTensorMeta::MutLocalTensorMeta()
    : MutTensorMeta(std::make_shared<const Shape>(), std::make_shared<const Stride>(),
                    kInvalidDataType),
      device_(Symbol<Device>()),
      storage_offset_(0) {}

MutLocalTensorMeta::MutLocalTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                                       Symbol<Device> device)
    : MutTensorMeta(shape, std::make_shared<const Stride>(*shape), dtype),
      device_(device),
      storage_offset_(0) {}

MutLocalTensorMeta::MutLocalTensorMeta(const std::shared_ptr<const Shape>& shape,
                                       const std::shared_ptr<const Stride>& stride, DataType dtype,
                                       Symbol<Device> device, int64_t storage_offset)
    : MutTensorMeta(shape, stride, dtype), device_(device), storage_offset_(storage_offset) {}

bool MutLocalTensorMeta::operator==(const MutLocalTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && *this->device() == *other.device() && this->stride() == other.stride()
         && this->storage_offset() == other.storage_offset();
}

size_t MutLocalTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Device>()(*device()) ^ std::hash<Stride>()(stride()) ^ storage_offset();
}

bool GlobalTensorMeta::operator==(const GlobalTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->nd_sbp() == other.nd_sbp() && this->parallel_desc() == other.parallel_desc();
}

size_t GlobalTensorMeta::CalcHashValue() const {
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Symbol<NdSbp>>()(nd_sbp())
         ^ std::hash<Symbol<ParallelDesc>>()(parallel_desc());
}

bool IsContiguous(const Shape& shape, const Stride& stride) {
  if (!shape.is_initialized()) { return true; }
  return IsContiguous(ShapeView(shape), stride);
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

}  // namespace one
}  // namespace oneflow
