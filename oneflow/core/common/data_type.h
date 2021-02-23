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

#include <type_traits>
#if defined(WITH_CUDA)
#include <cuda_fp16.h>
#endif
#include "oneflow/core/common/fp16_data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.pb.h"

namespace oneflow {

#if defined(WITH_CUDA)
#define DEVICE_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU) \
  OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU)
#else
#define DEVICE_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU)
#endif

// Type Trait: IsFloating

template<typename T>
struct IsFloating : std::integral_constant<bool, false> {};

#define SPECIALIZE_TRUE_FLOATING(type_cpp, type_proto) \
  template<>                                           \
  struct IsFloating<type_cpp> : std::integral_constant<bool, true> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_TRUE_FLOATING, FLOATING_DATA_TYPE_SEQ);
#undef SPECIALIZE_TRUE_FLOATING

// Type Trait: IsIntegral

template<typename T>
struct IsIntegral : std::integral_constant<bool, false> {};

#define SPECIALIZE_TRUE_INTEGRAL(type_cpp, type_proto) \
  template<>                                           \
  struct IsIntegral<type_cpp> : std::integral_constant<bool, true> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_TRUE_INTEGRAL, INT_DATA_TYPE_SEQ);
#undef SPECIALIZE_TRUE_INTEGRAL

// Type Trait: GetDataType

template<typename T, typename T2 = void>
struct GetDataType;

template<>
struct GetDataType<void> : std::integral_constant<DataType, DataType::kChar> {};

#define SPECIALIZE_GET_DATA_TYPE(type_cpp, type_proto)                            \
  template<>                                                                      \
  struct GetDataType<type_cpp> : std::integral_constant<DataType, type_proto> {}; \
  inline type_cpp GetTypeByDataType(std::integral_constant<DataType, type_proto>) { return {}; }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_DATA_TYPE, ALL_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
#undef SPECIALIZE_GET_DATA_TYPE

template<typename T>
struct GetDataType<T, typename std::enable_if<IsFloat16<T>::value>::type>
    : std::integral_constant<DataType, DataType::kFloat16> {};

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

#define MAX_VAL_SEQ                          \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, CHAR_MAX)     \
  OF_PP_MAKE_TUPLE_SEQ(int16_t, SHRT_MAX)    \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, INT_MAX)     \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, LLONG_MAX)   \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t, UCHAR_MAX)   \
  OF_PP_MAKE_TUPLE_SEQ(uint16_t, USHRT_MAX)  \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, UINT_MAX)   \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, ULLONG_MAX) \
  OF_PP_MAKE_TUPLE_SEQ(float, FLT_MAX)       \
  OF_PP_MAKE_TUPLE_SEQ(double, DBL_MAX)

#define MIN_VAL_SEQ                        \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, CHAR_MIN)   \
  OF_PP_MAKE_TUPLE_SEQ(int16_t, SHRT_MIN)  \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, INT_MIN)   \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, LLONG_MIN) \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t, 0)         \
  OF_PP_MAKE_TUPLE_SEQ(uint16_t, 0)        \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, 0)        \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, 0)        \
  OF_PP_MAKE_TUPLE_SEQ(float, -FLT_MAX)    \
  OF_PP_MAKE_TUPLE_SEQ(double, -DBL_MAX)

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

template<DeviceType, typename T>
struct DevDType {
  typedef T type;
};

#if defined(WITH_CUDA)
template<>
struct DevDType<DeviceType::kGPU, float16> {
  static_assert(sizeof(float16) == sizeof(half), "sizeof(float16) != sizeof(half)");
  typedef half type;
};
#endif

// Func

bool IsIntegralDataType(DataType data_type);
bool IsFloatingDataType(DataType data_type);
bool IsPODDataType(DataType data_type);
bool IsIndexDataType(DataType data_type);
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
