#include "oneflow/core/common/data_type.h"

namespace oneflow {

const DataType GetDataType<void>::val = DataType::kChar;

#define MACRO_PAIR(type_cpp, type_proto) \
  const DataType GetDataType<type_cpp>::val = type_proto;
ALL_DATA_TYPE_PAIR();
#undef MACRO_PAIR

#define MACRO_PAIR(type_cpp, type_proto) \
  const size_t GetSizeOf<type_proto>::val = sizeof(type_cpp);
ALL_DATA_TYPE_PAIR();
#undef MACRO_PAIR

}  // namespace oneflow
