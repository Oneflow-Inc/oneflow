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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {

Maybe<DType> DType::GetDTypeByDataType(const DataType& data_type) {
  switch (data_type) {
#define MAKE_DATA_TYPE_OBJ(data_type)       \
  case OF_PP_CAT(DataType::k, data_type): { \
    return data_type();                     \
  }
    OF_PP_FOR_EACH_TUPLE(MAKE_DATA_TYPE_OBJ, DTYPE_SEQ)
#undef MAKE_DATA_TYPE_OBJ
    default: { OF_UNIMPLEMENTED(); }
  }
  OF_UNIMPLEMENTED();
  return std::shared_ptr<DType>();
}

Maybe<DType> DType::InvalidDataType() {
  static std::shared_ptr<DType> invalid_dtype = std::make_shared<DType>(
      DataType::kInvalidDataType, "oneflow.invalid_data_type", false, false, false);
  return invalid_dtype;
}

Maybe<DType> DType::Char() {
  static std::shared_ptr<DType> char_dtype =
      std::make_shared<DType>(DataType::kChar, "oneflow.char", false, false, false);
  return char_dtype;
}

Maybe<DType> DType::Float16() {
  static std::shared_ptr<DType> float16_dtype =
      std::make_shared<DType>(DataType::kFloat16, "oneflow.float16", true, true, false);
  return float16_dtype;
}

Maybe<DType> DType::Float() {
  static std::shared_ptr<DType> float_dtype =
      std::make_shared<DType>(DataType::kFloat, "oneflow.float32", true, true, false);
  return float_dtype;
}

Maybe<DType> DType::Double() {
  static std::shared_ptr<DType> double_dtype =
      std::make_shared<DType>(DataType::kDouble, "oneflow.float64", true, true, false);
  return double_dtype;
}

Maybe<DType> DType::Int8() {
  static std::shared_ptr<DType> int8_dtype =
      std::make_shared<DType>(DataType::kInt8, "oneflow.int8", true, false, false);
  return int8_dtype;
}

Maybe<DType> DType::Int32() {
  static std::shared_ptr<DType> int32_dtype =
      std::make_shared<DType>(DataType::kInt32, "oneflow.int32", true, false, false);
  return int32_dtype;
}

Maybe<DType> DType::Int64() {
  static std::shared_ptr<DType> int64_dtype =
      std::make_shared<DType>(DataType::kInt64, "oneflow.int64", true, false, false);
  return int64_dtype;
}

Maybe<DType> DType::UInt8() {
  static std::shared_ptr<DType> uint8_dtype =
      std::make_shared<DType>(DataType::kUInt8, "oneflow.uint8", false, false, false);
  return uint8_dtype;
}

Maybe<DType> DType::OFRecord() {
  static std::shared_ptr<DType> record_dtype =
      std::make_shared<DType>(DataType::kOFRecord, "oneflow.of_record", false, false, false);
  return record_dtype;
}

Maybe<DType> DType::TensorBuffer() {
  static std::shared_ptr<DType> tensor_buffer_dtype = std::make_shared<DType>(
      DataType::kTensorBuffer, "oneflow.tensor_buffer", false, false, false);
  return tensor_buffer_dtype;
}

}  // namespace oneflow
