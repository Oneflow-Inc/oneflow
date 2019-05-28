#ifndef ONEFLOW_CORE_COMMON_DATA_TYPE_H_
#define ONEFLOW_CORE_COMMON_DATA_TYPE_H_

#include <half.hpp>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class OFRecord;
// SEQ

typedef half_float::half float16;

#define FLOATING_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat) \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define SIGNED_INT_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8)   \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define UNSIGNED_INT_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(uint8_t, DataType::kUInt8)

#define INT_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ

#define CHAR_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(char, DataType::kChar)

#define ARITHMETIC_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ         \
  INT_DATA_TYPE_SEQ

#define POD_DATA_TYPE_SEQ ARITHMETIC_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ
#define PB_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(OFRecord, DataType::kOFRecord)
#define ALL_DATA_TYPE_SEQ POD_DATA_TYPE_SEQ PB_DATA_TYPE_SEQ

#define FLOAT16_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(float16, DataType::kFloat16)

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

// Type Trait: IsFloat16
template<typename T>
struct IsFloat16 : std::integral_constant<bool, false> {};

#define SPECIALIZE_TRUE_FLOAT16(type_cpp, type_proto) \
  template<>                                          \
  struct IsFloat16<type_cpp> : std::integral_constant<bool, true> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_TRUE_FLOAT16, FLOAT16_DATA_TYPE_SEQ);
#undef SPECIALIZE_TRUE_FLOAT16

// Type Trait: GetDataType

template<typename T>
struct GetDataType;

template<>
struct GetDataType<void> : std::integral_constant<DataType, DataType::kChar> {};

#define SPECIALIZE_GET_DATA_TYPE(type_cpp, type_proto)                            \
  template<>                                                                      \
  struct GetDataType<type_cpp> : std::integral_constant<DataType, type_proto> {}; \
  inline type_cpp GetTypeByDataType(std::integral_constant<DataType, type_proto>) { return {}; }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_DATA_TYPE,
                     ALL_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
#undef SPECIALIZE_GET_DATA_TYPE

template<DataType type>
using DataTypeToType = decltype(GetTypeByDataType(std::integral_constant<DataType, type>{}));

// Type Trait: const var

#define TRAIT_CONST_VAR(var_name, var_val)                   \
  template<typename T>                                       \
  struct var_name##Val {                                     \
    static const T value;                                    \
  };                                                         \
  template<typename T>                                       \
  const T var_name##Val<T>::value = static_cast<T>(var_val); \
  template<typename T>                                       \
  struct var_name##Ptr {                                     \
    static const T* value;                                   \
  };                                                         \
  template<typename T>                                       \
  const T* var_name##Ptr<T>::value = &var_name##Val<T>::value;

TRAIT_CONST_VAR(Zero, 0);
TRAIT_CONST_VAR(One, 1);

#undef TRAIT_CONST_VAR

#if defined(__CUDACC__)
#define OF_DEVICE_FUNC __device__ __host__
#else
#define OF_DEVICE_FUNC
#endif

template<typename T>
OF_DEVICE_FUNC T GetZeroVal() {
  return static_cast<T>(0);
}

template<typename T>
OF_DEVICE_FUNC T GetOneVal() {
  return static_cast<T>(1);
}

template<typename T>
OF_DEVICE_FUNC T GetMinVal();

template<typename T>
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
  inline OF_DEVICE_FUNC T GetMaxVal<T>() { \
    return limit_value;                    \
  }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_MAX_VAL, MAX_VAL_SEQ);
#undef SPECIALIZE_MAX_VAL

#define SPECIALIZE_MIN_VAL(T, limit_value) \
  template<>                               \
  inline OF_DEVICE_FUNC T GetMinVal<T>() { \
    return limit_value;                    \
  }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_MIN_VAL, MIN_VAL_SEQ);
#undef SPECIALIZE_MIN_VAL

template<>
inline float16 GetZeroVal<float16>() {
  return float16(0.0);
}

template<>
inline float16 GetOneVal<float16>() {
  return float16(1.0);
}

template<>
inline float16 GetMaxVal<float16>() {
  return std::numeric_limits<float16>::max();
}

template<>
inline float16 GetMinVal<float16>() {
  return std::numeric_limits<float16>::lowest();
}

// Func

bool IsIntegralDataType(DataType data_type);
bool IsFloatingDataType(DataType data_type);
size_t GetSizeOfDataType(DataType data_type);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_H_
