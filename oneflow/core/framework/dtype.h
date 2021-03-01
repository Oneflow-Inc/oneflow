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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {

#define DTYPE_SEQ                       \
  OF_PP_MAKE_TUPLE_SEQ(InvalidDataType) \
  OF_PP_MAKE_TUPLE_SEQ(Char)            \
  OF_PP_MAKE_TUPLE_SEQ(Float16)         \
  OF_PP_MAKE_TUPLE_SEQ(Float)           \
  OF_PP_MAKE_TUPLE_SEQ(Double)          \
  OF_PP_MAKE_TUPLE_SEQ(Int8)            \
  OF_PP_MAKE_TUPLE_SEQ(Int32)           \
  OF_PP_MAKE_TUPLE_SEQ(Int64)           \
  OF_PP_MAKE_TUPLE_SEQ(UInt8)           \
  OF_PP_MAKE_TUPLE_SEQ(OFRecord)        \
  OF_PP_MAKE_TUPLE_SEQ(TensorBuffer)

class DType final {
 public:
  DType(const DataType& data_type, const std::string& name, bool is_signed, bool is_floating_point,
        bool is_complex)
      : data_type_(data_type),
        name_(name),
        is_signed_(is_signed),
        is_floating_point_(is_floating_point),
        is_complex_(is_complex) {}

  DataType data_type() const { return data_type_; }
  bool is_signed() const { return is_signed_; }
  bool is_complex() const { return is_complex_; }
  bool is_floating_point() const { return is_floating_point_; }
  std::string name() const { return name_; }

  static Maybe<DType> GetDTypeByDataType(const DataType&);

#define DEFINE_GET_DATA_TYPE_FUNCTION(data_type) static Maybe<DType> data_type();
  OF_PP_FOR_EACH_TUPLE(DEFINE_GET_DATA_TYPE_FUNCTION, DTYPE_SEQ)
#undef DEFINE_GET_DATA_TYPE_FUNCTION

 private:
  const DataType data_type_;
  const std::string name_;
  const bool is_signed_;
  const bool is_floating_point_;
  const bool is_complex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_DTYPE_H_
