#include "oneflow/core/common/data_type.h"

namespace oneflow {

bool IsIntegralDataType(DataType data_type) {
  switch (data_type) {
#define INTERGRAL_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(INTERGRAL_CASE, INT_DATA_TYPE_SEQ)
    default: return false;
  }
}
bool IsFloatingDataType(DataType data_type) {
  switch (data_type) {
#define FLOATING_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(FLOATING_CASE, FLOATING_DATA_TYPE_SEQ)
    default: return false;
  }
}

size_t GetSizeOfDataType(DataType data_type) {
  switch (data_type) {
#define MAKE_CASE(type_cpp, type_proto) \
  case type_proto: return sizeof(type_cpp);
    OF_PP_FOR_EACH_TUPLE(MAKE_CASE,
                         ALL_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
    default: UNIMPLEMENTED();
  }
}

}  // namespace oneflow
