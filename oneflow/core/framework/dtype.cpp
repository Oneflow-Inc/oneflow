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
#include "oneflow/core/framework/dtype.h"

namespace oneflow {

bool DType::is_floating_point() const {
  switch (oneflow_proto_dtype()) {
    case DataType::kFloat16:
    case DataType::kFloat:
    case DataType::kDouble: return true;
    default: return false;
  }
  return false;
}

std::shared_ptr<DType> Char() {
  static std::shared_ptr<DType> char_dtype = std::make_shared<DType>(DataType::kChar);
  return char_dtype;
}

std::shared_ptr<DType> Float16() {
  static std::shared_ptr<DType> float16_dtype = std::make_shared<DType>(DataType::kFloat16);
  return float16_dtype;
}

std::shared_ptr<DType> Float() {
  static std::shared_ptr<DType> float_dtype = std::make_shared<DType>(DataType::kFloat);
  return float_dtype;
}

std::shared_ptr<DType> Double() {
  static std::shared_ptr<DType> double_dtype = std::make_shared<DType>(DataType::kDouble);
  return double_dtype;
}

std::shared_ptr<DType> Int8() {
  static std::shared_ptr<DType> int8_dtype = std::make_shared<DType>(DataType::kInt8);
  return int8_dtype;
}

std::shared_ptr<DType> Int32() {
  static std::shared_ptr<DType> int32_dtype = std::make_shared<DType>(DataType::kInt32);
  return int32_dtype;
}

std::shared_ptr<DType> Int64() {
  static std::shared_ptr<DType> int64_dtype = std::make_shared<DType>(DataType::kInt64);
  return int64_dtype;
}

std::shared_ptr<DType> UInt8() {
  static std::shared_ptr<DType> uint8_dtype = std::make_shared<DType>(DataType::kUInt8);
  return uint8_dtype;
}

std::shared_ptr<DType> RecordDType() {
  static std::shared_ptr<DType> record_dtype = std::make_shared<DType>(DataType::kOFRecord);
  return record_dtype;
}

std::shared_ptr<DType> TensorBufferDType() {
  static std::shared_ptr<DType> tensor_buffer_dtype = std::make_shared<DType>(DataType::kTensorBuffer);
  return tensor_buffer_dtype;
}

}  // namespace oneflow
