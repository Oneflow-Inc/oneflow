#ifndef ONEFLOW_CORE_RECORD_RECORD_H_
#define ONEFLOW_CORE_RECORD_RECORD_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

#define ENCODE_CASE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(EncodeCase::kRaw) \
  OF_PP_MAKE_TUPLE_SEQ(EncodeCase::kJpeg)
#define ENCODE_CASE_DATA_TYPE_SEQ_PRODUCT                      \
  OF_PP_SEQ_PRODUCT(ENCODE_CASE_SEQ, ARITHMETIC_DATA_TYPE_SEQ) \
  OF_PP_SEQ_PRODUCT((EncodeCase::kProtobuf), PB_LIST_DATA_TYPE_SEQ)

#define PB_LIST_SIZE_SHIFT_SEQ (8)(16)(24)

#define PB_LIST_TYPE_PB_LIST_FIELD_SEQ          \
  OF_PP_MAKE_TUPLE_SEQ(BytesList, bytes_list)   \
  OF_PP_MAKE_TUPLE_SEQ(FloatList, float_list)   \
  OF_PP_MAKE_TUPLE_SEQ(DoubleList, double_list) \
  OF_PP_MAKE_TUPLE_SEQ(Int32List, int32_list)

#define MAKE_RECORD_DATA_TYPE_PB_LIST_FIELD_SEQ(pair, shift)                             \
  OF_PP_MAKE_TUPLE_SEQ(OF_PP_CAT(OF_PP_PAIR_FIRST(pair), shift),                         \
                       OF_PP_CAT(DataType::k, OF_PP_CAT(OF_PP_PAIR_FIRST(pair), shift)), \
                       OF_PP_PAIR_SECOND(pair))

#define PB_LIST_DATA_TYPE_PB_LIST_FIELD_SEQ                                 \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_RECORD_DATA_TYPE_PB_LIST_FIELD_SEQ, \
                                   PB_LIST_TYPE_PB_LIST_FIELD_SEQ, PB_LIST_SIZE_SHIFT_SEQ)

template<typename T>
inline void CheckRecordValue(const T& data) {
  // do nothing
}

#define SPECIALIZE_CHECK_RECORD_VALUE(type_prefix, shift)                            \
  template<>                                                                         \
  inline void CheckRecordValue<type_prefix##shift>(const type_prefix##shift& data) { \
    CHECK_EQ(data.value().value_size(), 1);                                          \
    CHECK_LE(data.value().value(0).size(), static_cast<int32_t>(1) << shift);        \
  }
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SPECIALIZE_CHECK_RECORD_VALUE, (BytesList),
                                 PB_LIST_SIZE_SHIFT_SEQ);
#undef SPECIALIZE_CHECK_RECORD_VALUE

#define SPECIALIZE_CHECK_RECORD_VALUE(type_prefix, shift)                            \
  template<>                                                                         \
  inline void CheckRecordValue<type_prefix##shift>(const type_prefix##shift& data) { \
    CHECK_LE(data.value().value_size(), static_cast<int32_t>(1) << shift);           \
  }
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SPECIALIZE_CHECK_RECORD_VALUE, (FloatList)(DoubleList)(Int32List),
                                 PB_LIST_SIZE_SHIFT_SEQ);
#undef SPECIALIZE_CHECK_RECORD_VALUE
}

#endif  // ONEFLOW_CORE_RECORD_RECORD_H_
