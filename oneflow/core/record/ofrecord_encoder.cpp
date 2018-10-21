#include "oneflow/core/record/ofrecord_encoder.h"
#include "oneflow/core/record/ofrecord_raw_encoder.h"
#include "oneflow/core/record/ofrecord_jpeg_encoder.h"
#include "oneflow/core/record/ofrecord_protobuf_encoder.h"
#include "oneflow/core/record/ofrecord_bytes_list_encoder.h"

namespace oneflow {

OFRecordEncoderIf* GetOFRecordEncoder(EncodeCase encode_case, DataType data_type) {
  static const HashMap<std::string, OFRecordEncoderIf*> obj = {
#define MAKE_ENTRY(et, dt) \
  {GetHashKey(et, OF_PP_PAIR_SECOND(dt)), new OFRecordEncoderImpl<et, OF_PP_PAIR_FIRST(dt)>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ENCODE_CASE_DATA_TYPE_SEQ_PRODUCT)};
  return obj.at(GetHashKey(encode_case, data_type));
}

}  // namespace oneflow
