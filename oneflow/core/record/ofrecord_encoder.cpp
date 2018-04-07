#include "oneflow/core/record/ofrecord_encoder.h"
#include "oneflow/core/record/ofrecord_raw_encoder.h"

namespace oneflow {

template<EncodeCase encode_case, typename T>
void OFRecordEncoder<encode_case, T>::EncodeOneFieldToOneRecord(
    DeviceCtx* ctx, int64_t record_id, const Blob* blob,
    const std::string& field_name, OFRecord& record) const {
  if (record.feature().find(field_name) != record.feature().end()) {
    LOG(FATAL) << "Field " << field_name << " found repeatedly in OfRecord";
  }
  int64_t one_col_elem_num = blob->shape().Count(1);
  Feature& feature = record.mutable_feature()->at(field_name);
  const T* in_dptr = blob->dptr<T>() + record_id * one_col_elem_num;
  EncodeOneCol(ctx, in_dptr, feature, field_name, one_col_elem_num);
}

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
