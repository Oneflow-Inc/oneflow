#ifndef ONEFLOW_CORE_REGISTER_DATA_TYPE_H_
#define ONEFLOW_CORE_REGISTER_DATA_TYPE_H_

#include "oneflow/core/register/blob_desc.pb.h"

namespace oneflow {

#define DATA_TYPE_MAP            \
  X(float, DataType::kFloat)     \
  X(double, DataType::kDouble)   \
  X(int8_t, DataType::kInt8)     \
  X(int16_t, DataType::kInt16)   \
  X(int32_t, DataType::kInt32)   \
  X(int64_t, DataType::kInt64)   \
  X(uint8_t, DataType::kUInt8)   \
  X(uint16_t, DataType::kUInt16) \
  X(uint32_t, DataType::kUInt32) \
  X(uint64_t, DataType::kUInt64) \
  X(char, DataType::kChar)

template<typename T>
struct GetDataType;

template<>
struct GetDataType<void> {
  static const DataType val = DataType::kChar;
};

#define X(type_cpp, type_proto)             \
  template<>                                \
  struct GetDataType<type_cpp> {            \
    static const DataType val = type_proto; \
  };
DATA_TYPE_MAP
#undef X

template<DataType data_type>
struct GetSizeOf;

#define X(type_cpp, type_proto)                 \
  template<>                                    \
  struct GetSizeOf<type_proto> {                \
    static const size_t val = sizeof(type_cpp); \
  };
DATA_TYPE_MAP
#undef X

inline size_t GetSizeOfDataType(DataType data_type) {
  static const HashMap<int, size_t> data_type2size = {
#define X(type_cpp, type_proto) {type_proto, sizeof(type_cpp)},
      DATA_TYPE_MAP
#undef X
  };
  return data_type2size.at(data_type);
}

#undef DATA_TYPE_MAP

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_DATA_TYPE_H_
