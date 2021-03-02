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
#ifndef ONEFLOW_XRT_PARAMETER_H_
#define ONEFLOW_XRT_PARAMETER_H_

#include <string>

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

#include "oneflow/core/common/data_type.h"  // GetSizeOfDataType

namespace oneflow {
namespace xrt {

inline int SizeOf(const DataType& data_type) {
  /*
    switch (data_type) {
      case DataType::kChar:
      case DataType::kInt8:
      case DataType::kUInt8:
        return 1;
      case DataType::kFloat16:
        return 2;
      case DataType::kFloat:
      case DataType::kInt32:
        return 4;
      case DataType::kDouble:
      case DataType::kInt64:
        return 8;
      default: {
        LOG(FATAL) << "Invalid data type " << data_type;
        return 0;
      }
    }
  */
  return GetSizeOfDataType(data_type);
}

class Parameter {
 public:
  Parameter() = default;
  virtual ~Parameter() = default;

  Parameter(void* data, const Shape& shape, const DataType& data_type)
      : storage_(data), shape_(shape), data_type_(data_type) {
    byte_size_ = shape.elem_cnt() * SizeOf(data_type);
  }

  Parameter(const std::string& name, void* data, const Shape& shape, const DataType& data_type)
      : storage_(data), shape_(shape), data_type_(data_type), parameter_name_(name) {
    byte_size_ = shape.elem_cnt() * SizeOf(data_type);
  }

  void set_data(const void* data) { storage_ = const_cast<void*>(data); }

  template<typename T>
  void set_data(const T* data) {
    storage_ = const_cast<T*>(data);
  }

  void* data() const { return storage_; }

  template<typename T>
  T* data() const {
    return reinterpret_cast<T*>(storage_);
  }

  const std::string& name() const { return parameter_name_; }
  const Shape& shape() const { return shape_; }
  const DataType& data_type() const { return data_type_; }
  int64_t byte_size() const {
    return byte_size_ < 0 ? shape_.elem_cnt() * SizeOf(data_type_) : byte_size_;
  }

  void set_name(const std::string& name) { parameter_name_ = name; }
  void set_shape(const Shape& shape) { shape_ = shape; }
  void set_data_type(const DataType& data_type) { data_type_ = data_type; }
  void set_byte_size(int64_t byte_size) { byte_size_ = byte_size; }

 private:
  void* storage_ = nullptr;

  int64_t byte_size_ = -1;

  Shape shape_;
  DataType data_type_;
  std::string parameter_name_{""};
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_PARAMETER_H_
