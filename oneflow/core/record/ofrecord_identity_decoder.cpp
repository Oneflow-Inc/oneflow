#include "oneflow/core/record/ofrecord_identity_decoder.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

template<>
int32_t OFRecordDecoderImpl<EncodeCase::kIdentity, OFRecord>::GetColNumOfFeature(
    const Feature& feature, int64_t one_col_elem_num) const {
  return 1;
}

template<>
void OFRecordDecoderImpl<EncodeCase::kIdentity, OFRecord>::ReadOneCol(
    DeviceCtx* ctx, const Feature& feature, const BlobConf& blob_conf, int32_t col_id,
    OFRecord* out_dptr, int64_t one_col_elem_num,
    std::function<int32_t(void)> NextRandomInt) const {
  (*out_dptr->mutable_feature())[kOFRecordMapDefaultKey] = feature;
}

#define INSTANTIATE_OFRECORD_IDENTITY_DECODER(type_cpp, type_proto) \
  template class OFRecordDecoderImpl<EncodeCase::kIdentity, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_IDENTITY_DECODER, RECORD_DATA_TYPE_SEQ)

}  // namespace oneflow
