#include "oneflow/core/common/data_type.h"

namespace oneflow {

bool IsIntegral(DataType data_type) {
  switch (data_type) {
#define IS_INTERGRAL_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(IS_INTERGRAL_CASE, INT_DATA_TYPE_SEQ)
    default: return false;
  }
}
bool IsFloatingPoint(DataType data_type) {
  switch (data_type) {
#define IS_FLOATING_POINT_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(IS_FLOATING_POINT_CASE, FLOATING_DATA_TYPE_SEQ)
    default: return false;
  }
}

const DataType GetDataType<void>::val = DataType::kChar;

#define DEFINE_GET_DATA_TYPE_VAL(type_cpp, type_proto) \
  const DataType GetDataType<type_cpp>::val = type_proto;
OF_PP_FOR_EACH_TUPLE(DEFINE_GET_DATA_TYPE_VAL, ALL_DATA_TYPE_SEQ);

#define DEFINE_GET_SIZE_OF_VAL(type_cpp, type_proto) \
  const size_t GetSizeOf<type_proto>::val = sizeof(type_cpp);
OF_PP_FOR_EACH_TUPLE(DEFINE_GET_SIZE_OF_VAL, ALL_DATA_TYPE_SEQ);

}  // namespace oneflow
