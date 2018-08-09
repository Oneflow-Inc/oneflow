#include "oneflow/core/record/ofrecord_identity_decoder.h"
#include "oneflow/core/record/record.h"

namespace oneflow {

namespace {

template<typename T>
decltype(std::declval<T>().value()) GetFeatureDataList(const Feature& feature);

#define SPECIALIZE_GET_FEATURE_DATA_LIST(T, type_proto, data_list)                    \
  template<>                                                                          \
  decltype(std::declval<T>().value()) GetFeatureDataList<T>(const Feature& feature) { \
    return feature.data_list();                                                       \
  }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_FEATURE_DATA_LIST, FEATURE_DATA_TYPE_FEATURE_FIELD_SEQ);

}  // namespace

template<typename T>
int32_t OFRecordDecoderImpl<EncodeCase::kIdentity, T>::GetColNumOfFeature(
    const Feature& feature, int64_t one_col_elem_num) const {
  return 1;
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kIdentity, T>::ReadOneCol(
    DeviceCtx* ctx, const Feature& feature, const BlobConf& blob_conf, int32_t col_id, T* out_dptr,
    int64_t one_col_elem_num, std::function<int32_t(void)> NextRandomInt) const {
  *out_dptr->mutable_value() = GetFeatureDataList<T>(feature);
  CheckRecordValue<T>(*out_dptr);
}

#define INSTANTIATE_OFRECORD_IDENTITY_DECODER(type_cpp, type_proto) \
  template class OFRecordDecoderImpl<EncodeCase::kIdentity, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_IDENTITY_DECODER, FEATURE_DATA_TYPE_SEQ);

}  // namespace oneflow
