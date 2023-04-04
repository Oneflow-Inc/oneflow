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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {

bool IsBoolDataType(DataType data_type) {
  switch (data_type) {
#define BOOL_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(BOOL_CASE, BOOL_DATA_TYPE_SEQ)
    default: return false;
  }
#undef BOOL_CASE
}

bool IsIntegralDataType(DataType data_type) {
  switch (data_type) {
#define INTEGRAL_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(INTEGRAL_CASE, INT_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ)
    default: return false;
  }
#undef INTEGRAL_CASE
}
bool IsFloatingDataType(DataType data_type) {
  switch (data_type) {
#define FLOATING_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(FLOATING_CASE,
                         FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ BFLOAT16_DATA_TYPE_SEQ)
    default: return false;
  }
#undef FLOATING_CASE
}
bool IsHalfDataType(DataType data_type) {
  switch (data_type) {
#define HALF_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(HALF_CASE, FLOAT16_DATA_TYPE_SEQ BFLOAT16_DATA_TYPE_SEQ)
    default: return false;
  }
#undef HALF_CASE
}
bool IsComplexDataType(DataType data_type) {
  switch (data_type) {
#define COMPLEX_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(COMPLEX_CASE, COMPLEX_DATA_TYPE_SEQ)
    default: return false;
  }
#undef COMPLEX_CASE
}
bool IsTriviallyCopyableDataType(DataType data_type) {
  switch (data_type) {
#define TRIVIALLY_COPY_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(TRIVIALLY_COPY_CASE, TRIVIALLY_COPY_DATA_TYPE_SEQ)
    default: return false;
  }
#undef TRIVIALLY_COPY_CASE
}
bool IsIndexDataType(DataType data_type) {
  switch (data_type) {
#define INDEX_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(INDEX_CASE, INDEX_DATA_TYPE_SEQ)
    default: return false;
  }
#undef INDEX_CASE
}
bool IsSupportRequireGradDataType(DataType data_type) {
  switch (data_type) {
#define REQUIRE_GRAD_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(REQUIRE_GRAD_CASE,
                         FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ BFLOAT16_DATA_TYPE_SEQ)
    default: return false;
  }
#undef REQUIRE_GRAD_CASE
}
bool NotSupportBoxingDataType(DataType data_type) {
  switch (data_type) {
#define NO_BOXING_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(NO_BOXING_CASE, NO_BOXING_DATA_TYPE_SEQ)
    default: return false;
  }
#undef NO_BOXING_CASE
}

size_t GetSizeOfDataType(DataType data_type) {
  switch (data_type) {
    // 8-bit
    case kChar: return 1;
    case kInt8: return 1;
    case kUInt8: return 1;
    case kBool: return 1;

    // 16-bit
    case kInt16: return 2;
    case kUInt16: return 2;
    case kFloat16: return 2;
    case kBFloat16: return 2;

    // 32-bit
    case kInt32: return 4;
    case kUInt32: return 4;
    case kFloat: return 4;
    case kComplex32: return 4;

    // 64-bit
    case kInt64: return 8;
    case kUInt64: return 8;
    case kDouble: return 8;
    case kComplex64: return 8;

    // 128-bit
    case kInt128: return 16;
    case kUInt128: return 16;
    case kComplex128: return 16;

    // non pod
    case kOFRecord: return sizeof(OFRecord);
    case kTensorBuffer: return sizeof(TensorBuffer);
    default: LOG(FATAL) << "invalid data_type: " << DataType_Name(data_type);
  }
}

namespace {

void CheckDataType() {
  static_assert(sizeof(int8_t) == sizeof(char), "sizeof(int8_t) != sizeof(char)");
  static_assert(sizeof(int16_t) == sizeof(short), "sizeof(int16_t) != sizeof(short)");
  static_assert(sizeof(int32_t) == sizeof(int), "sizeof(int32_t) != sizeof(int)");
  static_assert(sizeof(int64_t) == sizeof(long long), "sizeof(int64_t) != sizeof(long long)");

#if defined(WITH_CUDA)

#define CHECK_DEVICE_FP16(get_val)                              \
  do {                                                          \
    float16 host_fp16 = get_val<float16>();                     \
    half device_fp16 = get_val<half>();                         \
    CHECK_EQ(*(uint16_t*)&host_fp16, *(uint16_t*)&device_fp16); \
  } while (0)

  CHECK_DEVICE_FP16(GetZeroVal);
  CHECK_DEVICE_FP16(GetOneVal);
  CHECK_DEVICE_FP16(GetMaxVal);
  CHECK_DEVICE_FP16(GetMinVal);
#undef CHECK_DEVICE_FP16

#endif

#define CHECK_MAX_VAL(T, limit_value) CHECK_EQ(GetMaxVal<T>(), std::numeric_limits<T>::max());
  OF_PP_FOR_EACH_TUPLE(CHECK_MAX_VAL, MAX_VAL_SEQ);
#undef CHECK_MAX_VAL

#define CHECK_MIN_VAL(T, limit_value) CHECK_EQ(GetMinVal<T>(), std::numeric_limits<T>::lowest());
  OF_PP_FOR_EACH_TUPLE(CHECK_MIN_VAL, MIN_VAL_SEQ);
#undef CHECK_MIN_VAL
}

COMMAND(CheckDataType());

}  // namespace

}  // namespace oneflow
