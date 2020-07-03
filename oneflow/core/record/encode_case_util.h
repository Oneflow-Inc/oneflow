#ifndef ONEFLOW_CORE_RECORD_ENCODE_CASE_UTIL_H_
#define ONEFLOW_CORE_RECORD_ENCODE_CASE_UTIL_H_

#include "oneflow/core/common/data_type.h"

namespace oneflow {

//  encode case
#define ENCODE_CASE_DATA_TYPE_SEQ_PRODUCT                                            \
  OF_PP_SEQ_PRODUCT((EncodeCase::kJpeg), ARITHMETIC_DATA_TYPE_SEQ)                   \
  OF_PP_SEQ_PRODUCT((EncodeCase::kRaw), ARITHMETIC_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ) \
  OF_PP_SEQ_PRODUCT((EncodeCase::kBytesList), ((char, DataType::kChar))((int8_t, DataType::kInt8)))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_ENCODE_CASE_UTIL_H_
