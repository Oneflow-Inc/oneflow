#include "oneflow/core/record/ofrecord_protobuf_decoder.h"

namespace oneflow {

namespace {

template<typename T>
decltype(std::declval<T>().value()) GetFeatureDataList(const Feature& feature);

#define SPECIALIZE_GET_PB_LIST_DATA_LIST(T, type_proto, data_list)                    \
  template<>                                                                          \
  decltype(std::declval<T>().value()) GetFeatureDataList<T>(const Feature& feature) { \
    return feature.data_list();                                                       \
  }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_PB_LIST_DATA_LIST, PB_LIST_DATA_TYPE_PB_LIST_FIELD_SEQ);

}  // namespace

template<typename T>
int32_t OFRecordDecoderImpl<EncodeCase::kProtobuf, T>::GetColNumOfFeature(
    const Feature& feature, int64_t one_col_elem_num) const {
  return 1;
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kProtobuf, T>::ReadOneCol(
    DeviceCtx* ctx, const Feature& feature, const BlobConf& blob_conf, int32_t col_id, T* out_dptr,
    int64_t one_col_elem_num, std::function<int32_t(void)> NextRandomInt) const {
  *out_dptr->mutable_value() = GetFeatureDataList<T>(feature);
  CheckPbListSize<T>(*out_dptr);
}

#define INSTANTIATE_OFRECORD_PROTOBUF_DECODER(type_cpp, type_proto) \
  template class OFRecordDecoderImpl<EncodeCase::kProtobuf, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_PROTOBUF_DECODER, PB_LIST_DATA_TYPE_SEQ);

}  // namespace oneflow
