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
  DType() : proto_dtype_(DataType::kInvalidDataType) {}
  DType(const DataType& proto_dtype) : proto_dtype_(proto_dtype) {}

  std::string ToString() const { return DataType_Name(proto_dtype_); }
  DataType oneflow_proto_dtype() const { return proto_dtype_; }
  bool is_signed() const { return proto_dtype_ != DataType::kUInt8; }
  bool is_complex() const { return false; }
  bool is_floating_point() const;

 private:
  DataType proto_dtype_;
};

std::shared_ptr<DType> Char();
std::shared_ptr<DType> Float16();
std::shared_ptr<DType> Float();

std::shared_ptr<DType> Double();
std::shared_ptr<DType> Int8();
std::shared_ptr<DType> Int32();

std::shared_ptr<DType> Int64();
std::shared_ptr<DType> UInt8();
std::shared_ptr<DType> RecordDType();
std::shared_ptr<DType> TensorBufferDType();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_DTYPE_H_
