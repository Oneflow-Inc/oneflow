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
#ifndef ONEFLOW_CORE_COMMON_DATA_TYPE_H_
#define ONEFLOW_CORE_COMMON_DATA_TYPE_H_

#include <cfloat>
#include <type_traits>
#if defined(WITH_CUDA)
#include <cuda_fp16.h>
#include <cuda.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#endif
#include "oneflow/core/common/bfloat16.h"
#include "oneflow/core/common/bfloat16_math.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.h"
#include <half.hpp>

namespace std {

// Extend numeric_limits<half> for the C++ standard library.
#ifdef WITH_CUDA

template<>
struct numeric_limits<half> {
  static constexpr int digits = std::numeric_limits<half_float::half>::digits;

  static constexpr half_float::half lowest() {
    return std::numeric_limits<half_float::half>::lowest();
  }

  static constexpr half_float::half max() { return std::numeric_limits<half_float::half>::max(); }
};

#endif  // WITH_CUDA

}  // namespace std

namespace oneflow {

namespace detail {

template<typename>
struct IsFloat16Helper : std::false_type {};

template<typename>
struct IsFloatingHelper : std::false_type {};

template<typename>
struct IsIntegralHelper : std::false_type {};

template<typename>
struct IsUnsignedIntegralHelper : std::false_type {};

}  // namespace detail

using float16 = half_float::half;

#define DEFINE_SPEC(Trait, Type, Value) \
  template<>                            \
  struct Trait<Type> : std::integral_constant<bool, Value> {};

// Type Trait: IsFloat16

DEFINE_SPEC(detail::IsFloat16Helper, float16, true)
#ifdef WITH_CUDA
DEFINE_SPEC(detail::IsFloat16Helper, half, true)
#endif  // WITH_CUDA

template<typename T>
struct IsFloat16
    : std::integral_constant<bool,
                             (detail::IsFloat16Helper<typename std::remove_cv<T>::type>::value)> {};

// Type Trait: IsFloating

#define SPECIALIZE_TRUE_FLOATING(type_cpp, type_proto) \
  DEFINE_SPEC(detail::IsFloatingHelper, type_cpp, true)
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_TRUE_FLOATING, FLOATING_DATA_TYPE_SEQ);
#undef SPECIALIZE_TRUE_FLOATING
DEFINE_SPEC(detail::IsFloatingHelper, float16, true)
#ifdef WITH_CUDA
DEFINE_SPEC(detail::IsFloatingHelper, half, true)
#endif  // WITH_CUDA

template<typename T>
struct IsFloating
    : std::integral_constant<bool,
                             (detail::IsFloatingHelper<typename std::remove_cv<T>::type>::value)> {
};

// Type Trait: IsIntegral

#define SPECIALIZE_TRUE_INTEGRAL(type_cpp, type_proto) \
  DEFINE_SPEC(detail::IsIntegralHelper, type_cpp, true)
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_TRUE_INTEGRAL, INT_DATA_TYPE_SEQ);
#undef SPECIALIZE_TRUE_INTEGRAL

template<typename T>
struct IsIntegral
    : std::integral_constant<bool,
                             (detail::IsIntegralHelper<typename std::remove_cv<T>::type>::value)> {
};

// Type Trait: IsUnsignedIntegral

#define SPECIALIZE_TRUE_INTEGRAL(type_cpp, type_proto) \
  DEFINE_SPEC(detail::IsUnsignedIntegralHelper, type_cpp, true)
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_TRUE_INTEGRAL, UNSIGNED_INT_DATA_TYPE_SEQ);
#undef SPECIALIZE_TRUE_INTEGRAL

template<typename T>
struct IsUnsignedIntegral
    : std::integral_constant<
          bool, (detail::IsUnsignedIntegralHelper<typename std::remove_cv<T>::type>::value)> {};

#undef DEFINE_SPEC

// Type Trait: GetDataType

template<typename T, typename T2 = void>
struct GetDataType;

template<>
struct GetDataType<void> : std::integral_constant<DataType, DataType::kChar> {};

#define SPECIALIZE_GET_DATA_TYPE(type_cpp, type_proto)                            \
  template<>                                                                      \
  struct GetDataType<type_cpp> : std::integral_constant<DataType, type_proto> {}; \
  inline type_cpp GetTypeByDataType(std::integral_constant<DataType, type_proto>) { return {}; }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_DATA_TYPE,
                     ALL_DATA_TYPE_SEQ UNSIGNED_INT32_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
                         BFLOAT16_DATA_TYPE_SEQ COMPLEX_DATA_TYPE_SEQ UNSIGNED_INT64_DATA_TYPE_SEQ);
#undef SPECIALIZE_GET_DATA_TYPE

template<typename T>
struct GetDataType<T, typename std::enable_if<IsFloat16<T>::value>::type>
    : std::integral_constant<DataType, DataType::kFloat16> {};

#if CUDA_VERSION >= 11000
template<>
struct GetDataType<nv_bfloat16> : std::integral_constant<DataType, DataType::kBFloat16> {};
#endif

template<DataType type>
using DataTypeToType = decltype(GetTypeByDataType(std::integral_constant<DataType, type>{}));

#if defined(__CUDACC__)
#define OF_DEVICE_FUNC __device__ __host__ __forceinline__
#else
#define OF_DEVICE_FUNC inline
#endif

template<typename T, typename std::enable_if<!IsFloat16<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T GetZeroVal() {
  return static_cast<T>(0);
}

template<typename T, typename std::enable_if<!IsFloat16<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T GetOneVal() {
  return static_cast<T>(1);
}

template<typename T, typename std::enable_if<!IsFloat16<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T GetMinVal();

template<typename T, typename std::enable_if<!IsFloat16<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T GetMaxVal();

#ifdef __APPLE__
#define APPLE_MAX_VAL_SEQ OF_PP_MAKE_TUPLE_SEQ(unsigned long, ULONG_MAX)
#else
#define APPLE_MAX_VAL_SEQ
#endif

#define MAX_VAL_SEQ                          \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, INT8_MAX)     \
  OF_PP_MAKE_TUPLE_SEQ(int16_t, INT16_MAX)   \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, INT32_MAX)   \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, INT64_MAX)   \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t, UINT8_MAX)   \
  OF_PP_MAKE_TUPLE_SEQ(uint16_t, UINT16_MAX) \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, UINT32_MAX) \
  APPLE_MAX_VAL_SEQ                          \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, UINT64_MAX) \
  OF_PP_MAKE_TUPLE_SEQ(float, FLT_MAX)       \
  OF_PP_MAKE_TUPLE_SEQ(double, DBL_MAX)      \
  OF_PP_MAKE_TUPLE_SEQ(bool, true)

#ifdef __APPLE__
#define APPLE_MIN_VAL_SEQ OF_PP_MAKE_TUPLE_SEQ(unsigned long, 0)
#else
#define APPLE_MIN_VAL_SEQ
#endif

#define MIN_VAL_SEQ                        \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, INT8_MIN)   \
  OF_PP_MAKE_TUPLE_SEQ(int16_t, INT16_MIN) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, INT32_MIN) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, INT64_MIN) \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t, 0)         \
  OF_PP_MAKE_TUPLE_SEQ(uint16_t, 0)        \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, 0)        \
  APPLE_MIN_VAL_SEQ                        \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, 0)        \
  OF_PP_MAKE_TUPLE_SEQ(float, -FLT_MAX)    \
  OF_PP_MAKE_TUPLE_SEQ(double, -DBL_MAX)   \
  OF_PP_MAKE_TUPLE_SEQ(bool, false)

#define SPECIALIZE_MAX_VAL(T, limit_value) \
  template<>                               \
  OF_DEVICE_FUNC T GetMaxVal<T>() {        \
    return limit_value;                    \
  }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_MAX_VAL, MAX_VAL_SEQ);
#undef SPECIALIZE_MAX_VAL

#define SPECIALIZE_MIN_VAL(T, limit_value) \
  template<>                               \
  OF_DEVICE_FUNC T GetMinVal<T>() {        \
    return limit_value;                    \
  }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_MIN_VAL, MIN_VAL_SEQ);
#undef SPECIALIZE_MIN_VAL

template<typename T>
const T* GetZeroPtr() {
  static const T ret = GetZeroVal<T>();
  return &ret;
}

template<typename T>
const T* GetOnePtr() {
  static const T ret = GetOneVal<T>();
  return &ret;
}

template<typename T, typename std::enable_if<IsFloat16<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T GetZeroVal() {
  uint16_t ret = 0x0;  // Decimal: 0; Binary: 0 00000 0000000000
  return *(T*)&ret;
}

template<typename T, typename std::enable_if<IsFloat16<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T GetOneVal() {
  uint16_t ret = 0x3c00;  // Decimal: 15360; Binary: 0 01111 0000000000
  return *(T*)&ret;
}

template<typename T, typename std::enable_if<IsFloat16<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T GetMaxVal() {
  uint16_t ret = 0x7bff;  // Decimal: 31743; Binary: 0 11110 1111111111
  return *(T*)&ret;
}

template<typename T, typename std::enable_if<IsFloat16<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T GetMinVal() {
  uint16_t ret = 0xfbff;  // Decimal: 64511; Binary: 1 11110 1111111111
  return *(T*)&ret;
}

#if CUDA_VERSION >= 11000
template<>
OF_DEVICE_FUNC nv_bfloat16 GetMinVal<nv_bfloat16>() {
  uint16_t ret = 0xff7f;
  return *(nv_bfloat16*)&ret;
}
#endif  // CUDA_VERSION >= 11000

template<DeviceType, typename T>
struct DevDType {
  typedef T type;
};

#if defined(WITH_CUDA)
template<>
struct DevDType<DeviceType::kCUDA, float16> {
  static_assert(sizeof(float16) == sizeof(half), "sizeof(float16) != sizeof(half)");
  typedef half type;
};
#if CUDA_VERSION >= 11000
template<>
struct DevDType<DeviceType::kCUDA, bfloat16> {
  static_assert(sizeof(bfloat16) == sizeof(nv_bfloat16), "sizeof(bfloat16) != sizeof(nv_bfloat16)");
  typedef nv_bfloat16 type;
};
#endif  // CUDA_VERSION >= 11000
#endif  // defined(WITH_CUDA)

// Func

bool IsBoolDataType(DataType data_type);
bool IsIntegralDataType(DataType data_type);
bool IsFloatingDataType(DataType data_type);
bool IsHalfDataType(DataType data_type);
bool IsSupportRequireGradDataType(DataType data_type);
bool IsComplexDataType(DataType data_type);
bool IsTriviallyCopyableDataType(DataType data_type);
bool IsIndexDataType(DataType data_type);
bool NotSupportBoxingDataType(DataType data_type);
size_t GetSizeOfDataType(DataType data_type);

inline bool operator==(const OptInt64& lhs, const OptInt64& rhs) {
  return (lhs.has_value() && rhs.has_value() && lhs.value() == rhs.value())
         || (!lhs.has_value() && !rhs.has_value());
}

template<typename T>
void CheckDataType(DataType data_type) {
  LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                 && data_type != DataType::kChar && data_type != GetDataType<T>::value))
      << data_type << " " << GetDataType<T>::value;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_H_
