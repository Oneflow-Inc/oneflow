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
#include "half.hpp"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {

namespace {

template<typename T>
std::size_t GetDataTypeBytes() {
  return sizeof(T);
}

#define MAKE_DATA_TYPE_BYTES_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(
    std::size_t, GetDataTypeBytes, MAKE_DATA_TYPE_BYTES_SWITCH_ENTRY,
    MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ BFLOAT16_DATA_TYPE_SEQ));

class DTypeMeta final {
 public:
  DTypeMeta(const std::string& name, bool is_signed, bool is_integer, bool is_floating_point,
            bool is_complex)
      : name_(name),
        is_signed_(is_signed),
        is_integer_(is_integer),
        is_floating_point_(is_floating_point),
        is_complex_(is_complex) {}
  DTypeMeta(const DTypeMeta&) = default;
  DTypeMeta(DTypeMeta&) = default;
  ~DTypeMeta() = default;

  const std::string& name() const { return name_; }
  bool is_signed() const { return is_signed_; }
  bool is_integer() const { return is_integer_; }
  bool is_floating_point() const { return is_floating_point_; }
  bool is_complex() const { return is_complex_; }

 private:
  const std::string name_;
  const bool is_signed_;
  const bool is_integer_;
  const bool is_floating_point_;
  const bool is_complex_;
};

Maybe<const DTypeMeta&> DTypeMeta4DataType(DataType data_type) {
  static const HashMap<DataType, DTypeMeta> data_type2dtype_meta{
      {DataType::kInvalidDataType,
       DTypeMeta("oneflow.invalid_data_type", false, false, false, false)},
      {DataType::kChar, DTypeMeta("oneflow.char", false, false, false, false)},
      {DataType::kFloat16, DTypeMeta("oneflow.float16", true, false, true, false)},
      {DataType::kFloat, DTypeMeta("oneflow.float32", true, false, true, false)},
      {DataType::kDouble, DTypeMeta("oneflow.float64", true, false, true, false)},
      {DataType::kInt8, DTypeMeta("oneflow.int8", true, true, false, false)},
      {DataType::kInt16, DTypeMeta("oneflow.int16", true, true, false, false)},
      {DataType::kInt32, DTypeMeta("oneflow.int32", true, true, false, false)},
      {DataType::kInt64, DTypeMeta("oneflow.int64", true, true, false, false)},
      {DataType::kInt128, DTypeMeta("oneflow.int128", true, true, false, false)},
      {DataType::kUInt8, DTypeMeta("oneflow.uint8", false, true, false, false)},
      {DataType::kUInt16, DTypeMeta("oneflow.uint16", false, true, false, false)},
      {DataType::kUInt32, DTypeMeta("oneflow.uint32", false, true, false, false)},
      {DataType::kUInt64, DTypeMeta("oneflow.uint64", false, true, false, false)},
      {DataType::kUInt128, DTypeMeta("oneflow.uint128", false, true, false, false)},
      {DataType::kOFRecord, DTypeMeta("oneflow.of_record", false, false, false, false)},
      {DataType::kTensorBuffer, DTypeMeta("oneflow.tensor_buffer", false, false, false, false)},
      {DataType::kBFloat16, DTypeMeta("oneflow.bfloat16", true, false, true, false)},
      {DataType::kBool, DTypeMeta("oneflow.bool", false, false, false, false)},
      {DataType::kComplex32, DTypeMeta("oneflow.complex32", false, false, false, true)},
      {DataType::kComplex64, DTypeMeta("oneflow.complex64", false, false, false, true)},
      {DataType::kComplex128, DTypeMeta("oneflow.complex128", false, false, false, true)},
  };
  return MapAt(data_type2dtype_meta, data_type);
};

}  // namespace

Maybe<const Symbol<DType>&> DType::Get(DataType data_type) {
  static HashMap<DataType, const Symbol<DType>> data_type2dtype{
#define MAKE_ENTRY(data_type) {OF_PP_CAT(DataType::k, data_type), data_type()},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, DTYPE_SEQ)
#undef MAKE_ENTRY
  };
  return MapAt(data_type2dtype, data_type);
}

Maybe<size_t> DType::bytes() const {
  // DataType::OFRecord and DataType::TensorBuffer don't have fixed byte size
  if (data_type() == DataType::kInvalidDataType || data_type() == DataType::kOFRecord
      || data_type() == DataType::kTensorBuffer) {
    OF_UNIMPLEMENTED();
  }
  return SwitchGetDataTypeBytes(SwitchCase(data_type()));
}

bool DType::is_signed() const { return CHECK_JUST(DTypeMeta4DataType(data_type_)).is_signed(); }

bool DType::is_complex() const { return CHECK_JUST(DTypeMeta4DataType(data_type_)).is_complex(); }

/*
  The order of datatype is:
  0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19
  20 iv   c1   f4   f8   i1   i4   i8   u1   re   f2   bu   bf   b1   u4   u8   u16  i2   i16  cp4
  cp8  cp16 The priority order of datatype is: 0    1    2    3    4    5    6    7    8    9    10
  11    12   13   14   15    16    17     18   19   20 iv < b1 < u1 < c1 < i1 < i2 < u4 < i4 < u8 <
  i8 < u16 < i16 < f2 < f4 < f8 < cp4 < cp8 < cp16 < bf < re < bu.
*/
const int DType::priority_order[DataType_ARRAYSIZE] = {0,  /*kInvalid*/
                                                       3,  /*kChar*/
                                                       13, /*kFloat32*/
                                                       14, /*kDouble*/
                                                       4,  /*kInt8*/
                                                       7,  /*kInt32*/
                                                       9,  /*kInt64*/
                                                       2,  /*kUInt8*/
                                                       19, /*kOFRecord*/
                                                       12, /*kFloat16*/
                                                       20, /*kTensorBuffer*/
                                                       18, /*kBFloat16*/
                                                       1,  /*kBool*/
                                                       6,  /*kUint32*/
                                                       8,  /*kUint64*/
                                                       10, /*kUint128*/
                                                       5,  /*kInt16*/
                                                       11, /*kInt128*/
                                                       15, /*kComplex32*/
                                                       16, /*kComplex64*/
                                                       17 /*kComplex128*/};

bool DType::is_integer() const { return CHECK_JUST(DTypeMeta4DataType(data_type_)).is_integer(); }

bool DType::is_floating_point() const {
  return CHECK_JUST(DTypeMeta4DataType(data_type_)).is_floating_point();
}

const std::string& DType::name() const { return CHECK_JUST(DTypeMeta4DataType(data_type_)).name(); }

#define DEFINE_GET_DATA_TYPE_FUNCTION(data_type)                                   \
  const Symbol<DType>& DType::data_type() {                                        \
    static const auto& dtype = SymbolOf(DType(OF_PP_CAT(DataType::k, data_type))); \
    return dtype;                                                                  \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_GET_DATA_TYPE_FUNCTION, DTYPE_SEQ)
#undef DEFINE_GET_DATA_TYPE_FUNCTION

Symbol<DType> promoteTypes(const Symbol<DType> a, const Symbol<DType> b) {
  const Symbol<DType> iv = CHECK_JUST(DType::Get(DataType::kInvalidDataType));
  const Symbol<DType> c1 = CHECK_JUST(DType::Get(DataType::kChar));
  const Symbol<DType> f4 = CHECK_JUST(DType::Get(DataType::kFloat));
  const Symbol<DType> f8 = CHECK_JUST(DType::Get(DataType::kDouble));
  const Symbol<DType> i1 = CHECK_JUST(DType::Get(DataType::kInt8));
  const Symbol<DType> i4 = CHECK_JUST(DType::Get(DataType::kInt32));
  const Symbol<DType> i8 = CHECK_JUST(DType::Get(DataType::kInt64));
  const Symbol<DType> u1 = CHECK_JUST(DType::Get(DataType::kUInt8));
  const Symbol<DType> re = CHECK_JUST(DType::Get(DataType::kOFRecord));
  const Symbol<DType> f2 = CHECK_JUST(DType::Get(DataType::kFloat16));
  const Symbol<DType> bu = CHECK_JUST(DType::Get(DataType::kTensorBuffer));
  const Symbol<DType> bf = CHECK_JUST(DType::Get(DataType::kBFloat16));
  const Symbol<DType> b1 = CHECK_JUST(DType::Get(DataType::kBool));
  const Symbol<DType> u2 = CHECK_JUST(DType::Get(DataType::kUInt16));
  const Symbol<DType> u4 = CHECK_JUST(DType::Get(DataType::kUInt32));
  const Symbol<DType> u8 = CHECK_JUST(DType::Get(DataType::kUInt64));
  const Symbol<DType> u16 = CHECK_JUST(DType::Get(DataType::kUInt128));
  const Symbol<DType> i2 = CHECK_JUST(DType::Get(DataType::kInt16));
  const Symbol<DType> i16 = CHECK_JUST(DType::Get(DataType::kInt128));
  const Symbol<DType> cp4 = CHECK_JUST(DType::Get(DataType::kComplex32));
  const Symbol<DType> cp8 = CHECK_JUST(DType::Get(DataType::kComplex64));
  const Symbol<DType> cp16 = CHECK_JUST(DType::Get(DataType::kComplex128));

  /* It is consistent with data_type.proto(except kInvalidDataType, kOFRecord and kTensorBuffer)
    kInvalidDataType = 0;
    kChar = 1;
    kFloat = 2;
    kDouble = 3;
    kInt8 = 4;
    kInt32 = 5;
    kInt64 = 6;
    kUInt8 = 7;
    kOFRecord = 8;
    kFloat16 = 9;
    kTensorBuffer = 10;
    kBFloat16 = 11;
    kBool = 12;
    kUInt16 = 13;
    kUInt32 = 14;
    kUInt64 = 15;
    kUInt128 = 16;
    kInt16 = 17;
    kInt128 = 18;
    kComplex32 = 19;
    kComplex64 = 20;
    kComplex128 = 21;

    The priority order of datatype is:
    iv < b1 < u1 < c1 < i1 < u2 < i2 < u4 < i4 < u8 < i8 < u16 < i16 < f2 < f4 < f8 < cp4 < cp8 <
    cp16 < bf < re < bu.

    When int8 + uint8, it need to promote to int16, etc.
    But in int8 + uint128, we should promote to int256, but it is not exist, so we set as Invalid.

    The new DataType should be add in the end of proto, and the Loopup table should be maintained as
    right priority (author:zhengzekang).
  */

  // clang-format off
  static const Symbol<DType> _promoteTypesLookup[DataType_ARRAYSIZE][DataType_ARRAYSIZE] = {
      /*          iv   c1   f4   f8   i1   i4   i8   u1   re   f2   bu   bf   b1   u2   u4   u8   u16   i2   i16   cp4   cp8   cp16 */
      /* iv */   {iv,  c1,  f4,  f8,  i1,  i4,  i8,  u1,  re,  f2,  bu,  bf,  b1,  u2,  u4,  u8,  u16,  i2,  i16,  cp4,  cp8,  cp16},
      /* c1 */   {c1,  c1,  f4,  f8,  i1,  i4,  i8,  c1,  iv,  f2,  iv,  bf,  c1,  u2,  u4,  u8,  u16,  i2,  i16,  iv,   cp4,  cp16},
      /* f4 */   {f4,  f4,  f4,  f8,  f4,  f4,  f4,  f4,  iv,  f4,  iv,  bf,  f4,  f4,  f4,  f4,  f4,   f4,  f4,   iv,   cp4,  cp16},
      /* f8 */   {f8,  f8,  f8,  f8,  f8,  f8,  f8,  f8,  iv,  f8,  iv,  bf,  f8,  f8,  f8,  f8,  f8,   f8,  f8,   iv,   cp4,  cp16},
      /* i1 */   {i1,  i1,  f4,  f8,  i1,  i4,  i8,  i2,  iv,  f2,  iv,  bf,  i1,  i4,  i8,  i16, iv,   i2,  i16,  iv,   cp4,  cp16},
      /* i4 */   {i4,  i4,  f4,  f8,  i4,  i4,  i8,  i4,  iv,  f2,  iv,  bf,  i4,  i4,  i8,  i16, iv,   i4,  i16,  iv,   cp4,  cp16},
      /* i8 */   {i8,  i8,  f4,  f8,  i8,  i8,  i8,  i8,  iv,  f2,  iv,  bf,  i8,  i8,  i8,  i16, iv,   i8,  i16,  iv,   cp4,  cp16},
      /* u1 */   {u1,  c1,  f4,  f8,  i2,  i4,  i8,  u1,  iv,  f2,  iv,  bf,  u1,  u2,  u4,  u8,  u16,  i2,  i16,  iv,   cp4,  cp16},
      /* re */   {iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,   iv,  iv,   iv,   iv,   iv},
      /* f2 */   {f2,  f2,  f4,  f8,  f2,  f2,  f2,  f2,  iv,  f2,  iv,  bf,  f2,  f2,  f2,  f2,  iv,   f2,  f2,   iv,   cp4,  cp16},
      /* bu */   {iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  bu,  iv,  iv,  iv,  iv,  iv,  iv,   iv,  iv,   iv,   iv,   iv},
      /* bf */   {bf,  bf,  bf,  bf,  bf,  bf,  bf,  bf,  iv,  bf,  iv,  bf,  bf,  bf,  bf,  bf,  iv,   bf,  bf,   iv,   cp4,  cp16},
      /* b1 */   {b1,  c1,  f4,  f8,  i1,  i4,  i8,  u1,  iv,  f2,  iv,  bf,  b1,  u2,  u4,  u8,  u16,  i2,  i16,  iv,   cp4,  cp16},
      /* u2 */   {u2,  u2,  f4,  f8,  i4,  i4,  i8,  u2,  iv,  f2,  iv,  bf,  u2,  u2,  u4,  u8,  u16,  i4,  i16,  iv,   cp4,  cp16},
      /* u4 */   {u4,  u4,  f4,  f8,  i8,  i8,  i8,  u4,  iv,  f2,  iv,  bf,  u4,  u4,  u4,  u8,  u16,  i8,  i16,  iv,   cp4,  cp16},
      /* u8 */   {u8,  u8,  f4,  f8,  i16, i16, i16, u8,  iv,  f2,  iv,  bf,  u8,  u8,  u8,  u8,  u16,  i16, i16,  iv,   cp4,  cp16},
      /* u16 */  {u16, u16, f4,  f8,  iv,  iv,  iv,  u16, iv,  f2,  iv,  bf,  u16, u16, u16, u16, u16,  iv,  iv,   iv,   cp4,  cp16},
      /* i2 */   {i2,  i2,  f4,  f8,  i2,  i4,  i8,  i2,  iv,  f2,  iv,  bf,  i2,  i4,  i8,  i16, iv,   i2,  i16,  iv,   cp4,  cp16},
      /* i16 */  {i16, i16, f4,  f8,  i16, i16, i16, i16, iv,  f2,  iv,  bf,  i16, i16, i16, i16, iv,   i16, i16,  iv,   cp4,  cp16},
      /* cp4 */  {iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,  iv,   iv,  iv,   cp4,  cp8,  cp16},
      /* cp8 */  {cp8, cp8, cp8, cp8, cp8, cp8, cp8, cp8, iv,  cp8, iv,  cp8, cp8, cp8, cp8, cp8, cp8,  cp8, cp8,  cp8,  cp8,  cp16},
      /* cp16 */ {cp16,cp16,cp16,cp16,cp16,cp16,cp16,cp16,iv,  cp16,iv,  cp16,cp16,cp16,cp16,cp16,cp16, cp16,cp16, cp16, cp16, cp16}};
  // clang-format on
  return _promoteTypesLookup[static_cast<int>(a->data_type())][static_cast<int>(b->data_type())];
}

namespace {

std::mutex default_dtype_mutex;
Symbol<DType>* GetMutDefaultDTypeSymbol() {
  static Symbol<DType> default_dtype = CHECK_JUST(DType::Get(DataType::kFloat));
  return &default_dtype;
}

}  // namespace

Maybe<void> SetDefaultDType(const Symbol<DType>& dtype) {
  std::lock_guard<std::mutex> lock(default_dtype_mutex);
  CHECK_OR_RETURN(dtype->is_floating_point())
      << "only floating-point types are supported as the default type";
  *GetMutDefaultDTypeSymbol() = dtype;
  return Maybe<void>::Ok();
}

Symbol<DType> GetDefaultDType() {
  std::lock_guard<std::mutex> lock(default_dtype_mutex);
  return *GetMutDefaultDTypeSymbol();
}

}  // namespace oneflow
