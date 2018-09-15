#ifndef ONEFLOW_CORE_RECORD_ENCODE_H_
#define ONEFLOW_CORE_RECORD_ENCODE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

#define ENCODE_CASE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(EncodeCase::kRaw) \
  OF_PP_MAKE_TUPLE_SEQ(EncodeCase::kJpeg)
#define ENCODE_CASE_DATA_TYPE_SEQ_PRODUCT                      \
  OF_PP_SEQ_PRODUCT(ENCODE_CASE_SEQ, ARITHMETIC_DATA_TYPE_SEQ) \
  OF_PP_SEQ_PRODUCT((EncodeCase::kProtobuf), PB_LIST_DATA_TYPE_SEQ)
}

#endif  // ONEFLOW_CORE_RECORD_ENCODE_H_
