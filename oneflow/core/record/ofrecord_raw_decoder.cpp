#include "oneflow/core/record/ofrecord_raw_decoder.h"

namespace oneflow {

namespace {

template<typename T, typename U>
void FixInDptrThenCopyElem(DeviceCtx* ctx, const T* in_dptr, int32_t col_id,
                           int64_t one_col_elem_num, U* out_dptr) {
  in_dptr = in_dptr + col_id * one_col_elem_num;
  CopyElem(in_dptr, out_dptr, one_col_elem_num);
}

}  // namespace

template<typename T>
bool OFRecordDecoderImpl<EncodeCase::kRaw, T>::HasDim1ValidNumField(
    const EncodeConf& encode_conf) const {
  CHECK(encode_conf.has_raw());
  return encode_conf.raw().dim1_varying_length();
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kRaw, T>::SetDim1ValidNum(const Feature& feature,
                                                               Blob* out_blob,
                                                               int64_t dim0_idx) const {
  CHECK_GE(out_blob->static_shape().NumAxes(), 2);
  int64_t elem_num = 0;
  if (feature.has_bytes_list()) {
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    elem_num = feature.bytes_list().value(0).size();
  }
#define DEFINE_ONE_ELIF(PbT, CppT)                \
  else if (feature.has_##PbT##_list()) {          \
    elem_num = feature.PbT##_list().value_size(); \
  }
  DEFINE_ONE_ELIF(float, float)
  DEFINE_ONE_ELIF(double, double)
  DEFINE_ONE_ELIF(int32, int32_t)
#undef DEFINE_ONE_ELIF
  else {
    UNIMPLEMENTED();
  }
  CHECK_LE(elem_num, out_blob->static_shape().Count(1));
  CHECK_EQ(elem_num % out_blob->static_shape().Count(2), 0);
  out_blob->set_dim1_valid_num(dim0_idx, elem_num / out_blob->static_shape().Count(2));
}

template<typename T>
int32_t OFRecordDecoderImpl<EncodeCase::kRaw, T>::GetColNumOfFeature(
    const Feature& feature, int64_t one_col_elem_num) const {
  int64_t elem_num = 0;
  if (feature.has_bytes_list()) {
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    elem_num = feature.bytes_list().value(0).size();
  }
#define DEFINE_ONE_ELIF(PbT)                      \
  else if (feature.has_##PbT##_list()) {          \
    elem_num = feature.PbT##_list().value_size(); \
  }
  DEFINE_ONE_ELIF(float)
  DEFINE_ONE_ELIF(double)
  DEFINE_ONE_ELIF(int32)
#undef DEFINE_ONE_ELIF
  else {
    UNIMPLEMENTED();
  }
  CHECK_EQ(elem_num % one_col_elem_num, 0);
  CHECK_LE(elem_num, one_col_elem_num);
  return 1;
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kRaw, T>::ReadOneCol(
    DeviceCtx* ctx, const Feature& feature, const BlobConf& blob_conf, int32_t col_id, T* out_dptr,
    int64_t one_col_elem_num, std::function<int32_t(void)> NextRandomInt) const {
  // TODO: fix or remove col_id
  if (feature.has_bytes_list()) {
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    const auto& value0 = feature.bytes_list().value(0);
    auto in_dptr = reinterpret_cast<const int8_t*>(value0.c_str());
    one_col_elem_num = std::min<int64_t>(one_col_elem_num, value0.size());
    FixInDptrThenCopyElem<int8_t, T>(ctx, in_dptr, col_id, one_col_elem_num, out_dptr);
  }
#define DEFINE_ONE_ELIF(PbT, CppT)                                                     \
  else if (feature.has_##PbT##_list()) {                                               \
    const auto& list = feature.PbT##_list();                                           \
    const CppT* in_dptr = list.value().data();                                         \
    const int64_t padding_elem_num = blob_conf.encode_case().raw().auto_zero_padding() \
                                         ? one_col_elem_num - list.value_size()        \
                                         : 0;                                          \
    if (blob_conf.encode_case().raw().dim1_varying_length()                            \
        || blob_conf.encode_case().raw().auto_zero_padding()) {                        \
      CHECK_LE(list.value_size(), one_col_elem_num);                                   \
      one_col_elem_num = list.value_size();                                            \
    } else {                                                                           \
      CHECK_EQ(one_col_elem_num, list.value_size());                                   \
    }                                                                                  \
    FixInDptrThenCopyElem<CppT, T>(ctx, in_dptr, col_id, one_col_elem_num, out_dptr);  \
    if (padding_elem_num > 0) {                                                        \
      std::memset(out_dptr + one_col_elem_num, 0, padding_elem_num * sizeof(T));       \
    }                                                                                  \
  }
  DEFINE_ONE_ELIF(float, float)
  DEFINE_ONE_ELIF(double, double)
  DEFINE_ONE_ELIF(int32, int32_t)
#undef DEFINE_ONE_ELIF
  else {
    UNIMPLEMENTED();
  }
}

#define INSTANTIATE_OFRECORD_RAW_DECODER(type_cpp, type_proto) \
  template class OFRecordDecoderImpl<EncodeCase::kRaw, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_RAW_DECODER, ARITHMETIC_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ)

}  // namespace oneflow
