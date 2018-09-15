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

bool IsPbDataType(DataType data_type) {
  switch (data_type) {
#define PB_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(PB_CASE, PB_DATA_TYPE_SEQ)
    default: return false;
  }
}

size_t GetSizeOfDataType(DataType data_type) {
  switch (data_type) {
#define MAKE_CASE(type_cpp, type_proto) \
  case type_proto: return sizeof(type_cpp);
    OF_PP_FOR_EACH_TUPLE(MAKE_CASE, ALL_DATA_TYPE_SEQ);
    default: UNIMPLEMENTED();
  }
}

#define SPECIALIZE_CHECK_PB_LIST_SIZE(type_prefix, shift)                     \
  template<>                                                                  \
  void CheckPbListSize<type_prefix##shift>(const type_prefix##shift& data) {  \
    CHECK_EQ(data.value().value_size(), 1);                                   \
    CHECK_LE(data.value().value(0).size(), static_cast<int32_t>(1) << shift); \
  }
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SPECIALIZE_CHECK_PB_LIST_SIZE, (BytesList),
                                 PB_LIST_SIZE_SHIFT_SEQ);
#undef SPECIALIZE_CHECK_PB_LIST_SIZE

#define SPECIALIZE_CHECK_PB_LIST_SIZE(type_prefix, shift)                    \
  template<>                                                                 \
  void CheckPbListSize<type_prefix##shift>(const type_prefix##shift& data) { \
    CHECK_LE(data.value().value_size(), static_cast<int32_t>(1) << shift);   \
  }
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SPECIALIZE_CHECK_PB_LIST_SIZE, (FloatList)(DoubleList)(Int32List),
                                 PB_LIST_SIZE_SHIFT_SEQ);
#undef SPECIALIZE_CHECK_PB_LIST_SIZE

}  // namespace oneflow
