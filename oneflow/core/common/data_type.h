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

#define SIGNED_INT_DATA_TYPE_SEQ                \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define INT_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ

#define CHAR_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(char, DataType::kChar)

#define RECORD_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(OFRecord, DataType::kOFRecord)

#define ARITHMETIC_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ         \
  INT_DATA_TYPE_SEQ

#define POD_DATA_TYPE_SEQ ARITHMETIC_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ

#define ALL_DATA_TYPE_SEQ POD_DATA_TYPE_SEQ RECORD_DATA_TYPE_SEQ

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

template<DataType T>
struct IsRecordType : std::integral_constant<bool, false> {};

#define SPECIALIZE_TRUE_RECORD(type_cpp, type_proto) \
  template<>                                         \
  struct IsRecordType<type_proto> : std::integral_constant<bool, true> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_TRUE_RECORD, RECORD_DATA_TYPE_SEQ);
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
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_DATA_TYPE, ALL_DATA_TYPE_SEQ);
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

// Func

bool IsIntegralDataType(DataType data_type);
bool IsFloatingDataType(DataType data_type);
size_t GetSizeOfDataType(DataType data_type);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_H_
