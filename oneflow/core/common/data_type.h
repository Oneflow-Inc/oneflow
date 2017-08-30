#ifndef ONEFLOW_CORE_COMMON_DATA_TYPE_H_
#define ONEFLOW_CORE_COMMON_DATA_TYPE_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

#define FLOATING_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat) \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define SIGNED_INT_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8)   \
  OF_PP_MAKE_TUPLE_SEQ(int16_t, DataType::kInt16) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define UNSIGNED_INT_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t, DataType::kUInt8)   \
  OF_PP_MAKE_TUPLE_SEQ(uint16_t, DataType::kUInt16) \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, DataType::kUInt64)

#define CHAR_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(char, DataType::kChar)

#define INT_DATA_TYPE_SEQ  \
  SIGNED_INT_DATA_TYPE_SEQ \
  UNSIGNED_INT_DATA_TYPE_SEQ

#define ARITHMETIC_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ         \
  INT_DATA_TYPE_SEQ

#define ALL_DATA_TYPE_SEQ  \
  ARITHMETIC_DATA_TYPE_SEQ \
  CHAR_DATA_TYPE_SEQ

#define FOR_EACH_PAIR OF_PP_FOR_EACH_TUPLE
#define SEQ_PRODUCT_FOR_EACH_TUPLE OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE

bool IsIntegral(DataType data_type);
bool IsFloatingPoint(DataType data_type);

template<typename T>
struct GetDataType;

template<>
struct GetDataType<void> {
  static const DataType val;
};

#define DECLARE_GET_DATA_TYPE(type_cpp, type_proto) \
  template<>                                        \
  struct GetDataType<type_cpp> {                    \
    static const DataType val;                      \
  };
FOR_EACH_PAIR(DECLARE_GET_DATA_TYPE, ALL_DATA_TYPE_SEQ);

template<DataType data_type>
struct GetSizeOf;

#define DECLARE_GET_SIZE_OF(type_cpp, type_proto) \
  template<>                                      \
  struct GetSizeOf<type_proto> {                  \
    static const size_t val;                      \
  };
FOR_EACH_PAIR(DECLARE_GET_SIZE_OF, ALL_DATA_TYPE_SEQ);

inline size_t GetSizeOfDataType(DataType data_type) {
  static const HashMap<int, size_t> data_type2size = {
#define SIZE_OF_DATA_TYPE_ENTRY(type_cpp, type_proto) \
  {type_proto, sizeof(type_cpp)},
      FOR_EACH_PAIR(SIZE_OF_DATA_TYPE_ENTRY, ALL_DATA_TYPE_SEQ)};
  return data_type2size.at(data_type);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_H_
