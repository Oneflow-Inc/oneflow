#include "oneflow/core/record/ofrecord_encoder.h"
#include "oneflow/core/record/ofrecord_raw_encoder.h"

namespace oneflow {

OFRecordEncoderIf* GetOFRecordEncoder(EncodeCase encode_case,
                                      DataType data_type) {
  static const HashMap<std::string, OFRecordEncoderIf*> obj = {

#define MAKE_ENTRY(et, dt)                \
  {GetHashKey(et, OF_PP_PAIR_SECOND(dt)), \
   new OFRecordEncoderImpl<et, OF_PP_PAIR_FIRST(dt)>},

      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY,
                                       OF_PP_MAKE_TUPLE_SEQ(EncodeCase::kRaw),
                                       ARITHMETIC_DATA_TYPE_SEQ)

  };
  return obj.at(GetHashKey(encode_case, data_type));
}

}  // namespace oneflow
