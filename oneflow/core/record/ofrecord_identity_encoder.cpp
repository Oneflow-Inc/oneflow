#include "oneflow/core/record/ofrecord_identity_encoder.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

template<>
void OFRecordEncoderImpl<EncodeCase::kIdentity, OFRecord>::EncodeOneCol(
    DeviceCtx* ctx, const Blob* in_blob, int64_t in_offset, Feature& feature,
    const std::string& field_name, int64_t one_col_elem_num) const {
  const OFRecord& ofrecord = in_blob->dptr<OFRecord>()[in_offset];
  CHECK_EQ(ofrecord.feature_size(), 1);
  feature = ofrecord.feature().at(kOFRecordMapDefaultKey);
}

#define INSTANTIATE_OFRECORD_IDENTITY_ENCODER(type_cpp, type_proto) \
  template class OFRecordEncoderImpl<EncodeCase::kIdentity, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_IDENTITY_ENCODER, RECORD_DATA_TYPE_SEQ)

}  // namespace oneflow
