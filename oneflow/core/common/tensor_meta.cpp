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
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {
namespace one {

MutTensorMeta::MutTensorMeta()
    : TensorMeta(kInvalidDataType, MemoryFormat::kContiguous),
      shape_(std::make_shared<const Shape>()),
      stride_(std::make_shared<const Stride>()) {}

MutTensorMeta::MutTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                             MemoryFormat memory_format)
    : TensorMeta(dtype, memory_format),
      shape_(std::make_shared<const Shape>(*shape)),
      stride_(std::make_shared<const Stride>(*shape)) {}

MutTensorMeta::MutTensorMeta(const std::shared_ptr<const Shape>& shape,
                             const std::shared_ptr<const Stride>& stride, DataType dtype,
                             MemoryFormat memory_format)
    : TensorMeta(dtype, memory_format),
      shape_(std::make_shared<const Shape>(*shape)),
      stride_(std::make_shared<const Stride>(*stride)) {}

MutTensorMeta::MutTensorMeta(const Shape& shape, DataType dtype, MemoryFormat memory_format)
    : TensorMeta(dtype, memory_format),
      shape_(std::make_shared<const Shape>(shape)),
      stride_(std::make_shared<const Stride>(shape)) {}

MutTensorMeta::MutTensorMeta(const Shape& shape, const Stride& stride, DataType dtype,
                             MemoryFormat memory_format)
    : TensorMeta(dtype, memory_format),
      shape_(std::make_shared<const Shape>(shape)),
      stride_(std::make_shared<const Stride>(stride)) {}

bool MutTensorMeta::operator==(const MutTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->memory_format() == other.memory_format() && this->stride() == other.stride();
}

size_t MutTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return Hash(*shape_ptr(), dtype(), memory_format(), stride());
}

ConstTensorMeta::ConstTensorMeta()
    : TensorMeta(kInvalidDataType, MemoryFormat::kContiguous),
      shape_(SymbolOf(Shape())),
      stride_(SymbolOf(Stride())) {}

ConstTensorMeta::ConstTensorMeta(Symbol<Shape> shape, DataType dtype, MemoryFormat memory_format)
    : TensorMeta(dtype, memory_format), shape_(shape), stride_(SymbolOf(Stride(*shape))) {}

ConstTensorMeta::ConstTensorMeta(Symbol<Shape> shape, Symbol<Stride> stride, DataType dtype,
                                 MemoryFormat memory_format)
    : TensorMeta(dtype, memory_format), shape_(shape), stride_(stride) {}

bool ConstTensorMeta::operator==(const ConstTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->memory_format() == other.memory_format() && this->stride() == other.stride();
}

size_t ConstTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return Hash(*shape_ptr(), dtype(), memory_format(), stride());
}

LocalTensorMeta::LocalTensorMeta()
    : ConstTensorMeta(SymbolOf(Shape()), SymbolOf(Stride()), DataType::kInvalidDataType,
                      MemoryFormat::kContiguous),
      device_(Symbol<Device>()) {}

LocalTensorMeta::LocalTensorMeta(Symbol<Shape> shape, DataType dtype, MemoryFormat memory_format,
                                 Symbol<Device> device)
    : ConstTensorMeta(shape, SymbolOf(Stride(*shape)), dtype, memory_format), device_(device) {}

LocalTensorMeta::LocalTensorMeta(Symbol<Shape> shape, Symbol<Stride> stride, DataType dtype,
                                 MemoryFormat memory_format, Symbol<Device> device)
    : ConstTensorMeta(shape, stride, dtype, memory_format), device_(device) {}

LocalTensorMeta::LocalTensorMeta(Symbol<Shape> shape, Symbol<Stride> stride, DataType dtype,
                                 MemoryFormat memory_format, Symbol<Device> device,
                                 const bool is_view)
    : ConstTensorMeta(shape, stride, dtype, memory_format), device_(device), is_view_(is_view) {}

bool LocalTensorMeta::operator==(const LocalTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->memory_format() == other.memory_format() && this->device() == other.device()
         && this->stride() == other.stride();
}

size_t LocalTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return Hash(*shape_ptr(), dtype(), memory_format(), device(), stride());
}

MutLocalTensorMeta::MutLocalTensorMeta()
    : MutTensorMeta(std::make_shared<const Shape>(), std::make_shared<const Stride>(),
                    kInvalidDataType, MemoryFormat::kContiguous),
      device_(Symbol<Device>()) {}

MutLocalTensorMeta::MutLocalTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                                       MemoryFormat memory_format, Symbol<Device> device)
    : MutTensorMeta(shape, std::make_shared<const Stride>(*shape), dtype, memory_format),
      device_(device) {}

MutLocalTensorMeta::MutLocalTensorMeta(const std::shared_ptr<const Shape>& shape,
                                       const std::shared_ptr<const Stride>& stride, DataType dtype,
                                       MemoryFormat memory_format, Symbol<Device> device)
    : MutTensorMeta(shape, stride, dtype, memory_format), device_(device) {}

MutLocalTensorMeta::MutLocalTensorMeta(const Shape& shape, DataType dtype,
                                       MemoryFormat memory_format, Symbol<Device> device)
    : MutTensorMeta(shape, Stride(shape), dtype, memory_format), device_(device) {}

MutLocalTensorMeta::MutLocalTensorMeta(const Shape& shape, const Stride& stride, DataType dtype,
                                       MemoryFormat memory_format, Symbol<Device> device)
    : MutTensorMeta(shape, stride, dtype, memory_format), device_(device) {}

bool MutLocalTensorMeta::operator==(const MutLocalTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->memory_format() == other.memory_format() && *this->device() == *other.device()
         && this->stride() == other.stride();
}

size_t MutLocalTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return Hash(*shape_ptr(), dtype(), memory_format(), *device(), stride());
}

bool GlobalTensorMeta::operator==(const GlobalTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->memory_format() == other.memory_format() && this->nd_sbp() == other.nd_sbp()
         && this->parallel_desc() == other.parallel_desc();
}

size_t GlobalTensorMeta::CalcHashValue() const {
  return Hash(*shape_ptr(), dtype(), memory_format(), nd_sbp(), parallel_desc());
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
