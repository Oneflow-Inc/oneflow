#ifndef ONEFLOW_CORE_COMMON_DATA_TYPE_H_
#define ONEFLOW_CORE_COMMON_DATA_TYPE_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

#define FLOATING_DATA_TYPE_PAIR()     \
  MACRO_PAIR(float, DataType::kFloat) \
  MACRO_PAIR(double, DataType::kDouble)

#define SIGNED_INT_DATA_TYPE_PAIR()     \
  MACRO_PAIR(int8_t, DataType::kInt8)   \
  MACRO_PAIR(int16_t, DataType::kInt16) \
  MACRO_PAIR(int32_t, DataType::kInt32) \
  MACRO_PAIR(int64_t, DataType::kInt64)

#define UNSIGNED_INT_DATA_TYPE_PAIR()     \
  MACRO_PAIR(uint8_t, DataType::kUInt8)   \
  MACRO_PAIR(uint16_t, DataType::kUInt16) \
  MACRO_PAIR(uint32_t, DataType::kUInt32) \
  MACRO_PAIR(uint64_t, DataType::kUInt64)

#define ARITHMETIC_DATA_TYPE_PAIR() \
  FLOATING_DATA_TYPE_PAIR()         \
  SIGNED_INT_DATA_TYPE_PAIR()       \
  UNSIGNED_INT_DATA_TYPE_PAIR()

#define ALL_DATA_TYPE_PAIR()  \
  ARITHMETIC_DATA_TYPE_PAIR() \
  MACRO_PAIR(char, DataType::kChar)

template<typename T>
struct GetDataType;

template<>
struct GetDataType<void> {
  static const DataType val;
};

#define MACRO_PAIR(type_cpp, type_proto) \
  template<>                             \
  struct GetDataType<type_cpp> {         \
    static const DataType val;           \
  };
ALL_DATA_TYPE_PAIR();
#undef MACRO_PAIR

template<DataType data_type>
struct GetSizeOf;

#define MACRO_PAIR(type_cpp, type_proto) \
  template<>                             \
  struct GetSizeOf<type_proto> {         \
    static const size_t val;             \
  };
ALL_DATA_TYPE_PAIR();
#undef MACRO_PAIR

inline size_t GetSizeOfDataType(DataType data_type) {
  static const HashMap<int, size_t> data_type2size = {
#define MACRO_PAIR(type_cpp, type_proto) {type_proto, sizeof(type_cpp)},
      ALL_DATA_TYPE_PAIR()
#undef MACRO_PAIR
  };
  return data_type2size.at(data_type);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_H_
