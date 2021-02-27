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
#ifndef ONEFLOW_CORE_FRAMEWORK_DTYPE_H_
#define ONEFLOW_CORE_FRAMEWORK_DTYPE_H_

#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {

class DType final {
 public:
  DType(const DataType& proto_dtype, const std::string& name, bool is_signed,
        bool is_floating_point, bool is_complex)
      : proto_dtype_(proto_dtype),
        name_(name),
        is_signed_(is_signed),
        is_floating_point_(is_floating_point),
        is_complex_(is_complex) {}
  DType() : DType(DataType::kInvalidDataType, "oneflow.invalid_data_type", false, false, false) {}

  DataType oneflow_proto_dtype() const { return proto_dtype_; }
  bool is_signed() const { return is_signed_; }
  bool is_complex() const { return is_complex_; }
  bool is_floating_point() const { return is_floating_point_; }
  std::string name() const { return name_; }

  static std::shared_ptr<DType> GetDTypeByDataType(const DataType&);

  static std::shared_ptr<DType> Char();
  static std::shared_ptr<DType> Float16();
  static std::shared_ptr<DType> Float();

  static std::shared_ptr<DType> Double();
  static std::shared_ptr<DType> Int8();
  static std::shared_ptr<DType> Int32();

  static std::shared_ptr<DType> Int64();
  static std::shared_ptr<DType> UInt8();
  static std::shared_ptr<DType> OFRecordDType();
  static std::shared_ptr<DType> TensorBufferDType();

 private:
  const DataType proto_dtype_;
  const std::string name_;
  const bool is_signed_;
  const bool is_floating_point_;
  const bool is_complex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_DTYPE_H_
