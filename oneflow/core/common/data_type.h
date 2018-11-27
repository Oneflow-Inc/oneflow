#ifndef ONEFLOW_CORE_COMMON_DATA_TYPE_H_
#define ONEFLOW_CORE_COMMON_DATA_TYPE_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class OFRecord;
// SEQ

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

template<typename T>
struct GetDataType;

template<>
struct GetDataType<void> : std::integral_constant<DataType, DataType::kChar> {};

#define SPECIALIZE_GET_DATA_TYPE(type_cpp, type_proto)                            \
  template<>                                                                      \
  struct GetDataType<type_cpp> : std::integral_constant<DataType, type_proto> {}; \
  inline type_cpp GetTypeByDataType(std::integral_constant<DataType, type_proto>) { return {}; }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_DATA_TYPE, ALL_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ);
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

template<typename T>
struct MaxVal;
template<typename T>
struct MinVal;

#define TRAIT_LIMIT_VAL(max_or_min, T, limit_value)                                         \
  template<>                                                                                \
  struct max_or_min##Val<T> final {                                                         \
    static_assert(alignof(int) == alignof(int32_t), "int32_t should be exactly int");       \
    static_assert(alignof(long long) == alignof(int64_t), "int32_t should be exactly int"); \
    constexpr static T value = limit_value;                                                 \
  }

TRAIT_LIMIT_VAL(Max, int8_t, CHAR_MAX);
TRAIT_LIMIT_VAL(Max, int32_t, INT_MAX);
TRAIT_LIMIT_VAL(Max, uint32_t, UINT_MAX);
TRAIT_LIMIT_VAL(Max, int64_t, LLONG_MAX);
TRAIT_LIMIT_VAL(Max, uint64_t, ULLONG_MAX);
TRAIT_LIMIT_VAL(Max, float, FLT_MAX);
TRAIT_LIMIT_VAL(Max, double, DBL_MAX);

TRAIT_LIMIT_VAL(Min, int8_t, CHAR_MIN);
TRAIT_LIMIT_VAL(Min, int32_t, INT_MIN);
TRAIT_LIMIT_VAL(Min, uint32_t, 0);
TRAIT_LIMIT_VAL(Min, int64_t, LLONG_MIN);
TRAIT_LIMIT_VAL(Min, uint64_t, 0);
TRAIT_LIMIT_VAL(Min, float, -FLT_MAX);
TRAIT_LIMIT_VAL(Min, double, -DBL_MAX);

#undef TRAIT_LIMIT_VAL

// Func

bool IsIntegralDataType(DataType data_type);
bool IsFloatingDataType(DataType data_type);
size_t GetSizeOfDataType(DataType data_type);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_H_
