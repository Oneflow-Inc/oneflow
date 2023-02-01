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
#include "oneflow/core/common/symbol.h"

namespace oneflow {

#define DTYPE_SEQ                       \
  OF_PP_MAKE_TUPLE_SEQ(InvalidDataType) \
  OF_PP_MAKE_TUPLE_SEQ(Bool)            \
  OF_PP_MAKE_TUPLE_SEQ(Char)            \
  OF_PP_MAKE_TUPLE_SEQ(Float16)         \
  OF_PP_MAKE_TUPLE_SEQ(Float)           \
  OF_PP_MAKE_TUPLE_SEQ(Double)          \
  OF_PP_MAKE_TUPLE_SEQ(Int8)            \
  OF_PP_MAKE_TUPLE_SEQ(Int32)           \
  OF_PP_MAKE_TUPLE_SEQ(Int64)           \
  OF_PP_MAKE_TUPLE_SEQ(UInt8)           \
  OF_PP_MAKE_TUPLE_SEQ(OFRecord)        \
  OF_PP_MAKE_TUPLE_SEQ(TensorBuffer)    \
  OF_PP_MAKE_TUPLE_SEQ(BFloat16)        \
  OF_PP_MAKE_TUPLE_SEQ(UInt16)          \
  OF_PP_MAKE_TUPLE_SEQ(UInt32)          \
  OF_PP_MAKE_TUPLE_SEQ(UInt64)          \
  OF_PP_MAKE_TUPLE_SEQ(UInt128)         \
  OF_PP_MAKE_TUPLE_SEQ(Int16)           \
  OF_PP_MAKE_TUPLE_SEQ(Int128)          \
  OF_PP_MAKE_TUPLE_SEQ(Complex32)       \
  OF_PP_MAKE_TUPLE_SEQ(Complex64)       \
  OF_PP_MAKE_TUPLE_SEQ(Complex128)

class DType final {
 public:
  DType(const DType&) = default;
  DType(DType&&) = delete;
  explicit DType(DataType data_type) : data_type_(data_type) {}
  ~DType() = default;

  bool operator==(const DType& other) const { return this->data_type() == other.data_type(); }

  DataType data_type() const { return data_type_; }
  bool is_signed() const;
  bool is_complex() const;
  bool is_integer() const;
  bool is_floating_point() const;
  const std::string& name() const;
  Maybe<size_t> bytes() const;

  static Maybe<const Symbol<DType>&> Get(DataType);
  static const int priority_order[DataType_ARRAYSIZE];

#define DECLARE_GET_DATA_TYPE_FUNCTION(data_type) static const Symbol<DType>& data_type();
  OF_PP_FOR_EACH_TUPLE(DECLARE_GET_DATA_TYPE_FUNCTION, DTYPE_SEQ)
#undef DECLARE_GET_DATA_TYPE_FUNCTION

 private:
  DataType data_type_;
};

Symbol<DType> promoteTypes(const Symbol<DType> a, const Symbol<DType> b);

Maybe<void> SetDefaultDType(const Symbol<DType>& dtype);
Symbol<DType> GetDefaultDType();

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::DType> final {
  size_t operator()(const oneflow::DType& dtype) const {
    return static_cast<size_t>(dtype.data_type());
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_DTYPE_H_
