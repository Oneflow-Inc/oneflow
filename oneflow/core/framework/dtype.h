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
  OF_PP_MAKE_TUPLE_SEQ(BFloat16)

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
  bool is_floating_point() const;
  const std::string& name() const;
  Maybe<size_t> bytes() const;

  static Maybe<const Symbol<DType>&> Get(DataType);
  static const int priority_order[DataType::kMaxDataType];

#define DECLARE_GET_DATA_TYPE_FUNCTION(data_type) static const Symbol<DType>& data_type();
  OF_PP_FOR_EACH_TUPLE(DECLARE_GET_DATA_TYPE_FUNCTION, DTYPE_SEQ)
#undef DECLARE_GET_DATA_TYPE_FUNCTION

 private:
  DataType data_type_;
};

// static inline Symbol<DType> promoteTypes(const Symbol<DType> a, const Symbol<DType> b) {
//   const Symbol<DType> iv = CHECK_JUST(DType::Get(DataType::kInvalidDataType));
//   const Symbol<DType> c1 = CHECK_JUST(DType::Get(DataType::kChar));
//   const Symbol<DType> f4 = CHECK_JUST(DType::Get(DataType::kFloat));
//   const Symbol<DType> f8 = CHECK_JUST(DType::Get(DataType::kDouble));
//   const Symbol<DType> i1 = CHECK_JUST(DType::Get(DataType::kInt8));
//   const Symbol<DType> i4 = CHECK_JUST(DType::Get(DataType::kInt32));
//   const Symbol<DType> i8 = CHECK_JUST(DType::Get(DataType::kInt64));
//   const Symbol<DType> u1 = CHECK_JUST(DType::Get(DataType::kUInt8));
//   const Symbol<DType> re = CHECK_JUST(DType::Get(DataType::kOFRecord));
//   const Symbol<DType> f2 = CHECK_JUST(DType::Get(DataType::kFloat16));
//   const Symbol<DType> bu = CHECK_JUST(DType::Get(DataType::kTensorBuffer));
//   const Symbol<DType> bf = CHECK_JUST(DType::Get(DataType::kBFloat16));

//   /* It is consistent with data_type.proto(except kInvalidDataType, kOFRecord and kTensorBuffer)
//     kInvalidDataType = 0;
//     kChar = 1;
//     kFloat = 2;
//     kDouble = 3;
//     kInt8 = 4;
//     kInt32 = 5;
//     kInt64 = 6;
//     kUInt8 = 7;
//     kOFRecord = 8;
//     kFloat16 = 9;
//     kTensorBuffer = 10;
//     kBFloat16 = 11;

//     The priority order of datatype is:
//     iv < u1 < c1 < i1 < i4 < i8 < f2 < f4 < f8 < bf < re < bu.

//     The new DataType should be add in the end of proto, and the Loopup table should be maintained as
//     right priority (author:zhengzekang).
//   */
//   static const Symbol<DType> _promoteTypesLookup[DType::dtype_num][DType::dtype_num] = {
//       /*        iv  c1  f4  f8  i1  i4  i8  u1  re  f2  bu  bf */
//       /* iv */ {iv, c1, f4, f8, i1, i4, i8, u1, re, f2, bu, bf},
//       /* c1 */ {c1, c1, f4, f8, i1, i4, i8, c1, re, f2, bu, bf},
//       /* f4 */ {f4, f4, f4, f8, f4, f4, f4, f4, re, f4, bu, bf},
//       /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, re, f8, bu, bf},
//       /* i1 */ {i1, i1, f4, f8, i1, i4, i8, i1, re, f2, bu, bf},
//       /* i4 */ {i4, i4, f4, f8, i4, i4, i8, i4, re, f2, bu, bf},
//       /* i8 */ {i8, i8, f4, f8, i8, i8, i8, i8, re, f2, bu, bf},
//       /* u1 */ {u1, c1, f4, f8, i1, i4, i8, u1, re, f2, bu, bf},
//       /* re */ {re, re, re, re, re, re, re, re, re, re, bu, re},
//       /* f2 */ {f2, f2, f4, f8, f2, f2, f2, f2, re, f2, bu, bf},
//       /* bu */ {bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu},
//       /* bf */ {bf, bf, bf, bf, bf, bf, bf, bf, re, bf, bu, bf},
//   };

//   return _promoteTypesLookup[static_cast<int>(a->data_type())][static_cast<int>(b->data_type())];
// }

Symbol<DType> promoteTypes(const Symbol<DType> a, const Symbol<DType> b); 

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
