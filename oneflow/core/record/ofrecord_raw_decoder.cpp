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
  return elem_num / one_col_elem_num;
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kRaw, T>::ReadOneCol(
    DeviceCtx* ctx, const Feature& feature, const BlobConf& blob_conf,
    int32_t col_id, T* out_dptr, int64_t one_col_elem_num,
    std::function<int32_t(void)> NextRandomInt) const {
  if (feature.has_bytes_list()) {
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    auto in_dptr =
        reinterpret_cast<const int8_t*>(feature.bytes_list().value(0).c_str());
    FixInDptrThenCopyElem<int8_t, T>(ctx, in_dptr, col_id, one_col_elem_num,
                                     out_dptr);
  }
#define DEFINE_ONE_ELIF(PbT, CppT)                                         \
  else if (feature.has_##PbT##_list()) {                                   \
    const CppT* in_dptr = feature.PbT##_list().value().data();             \
    FixInDptrThenCopyElem<CppT, T>(ctx, in_dptr, col_id, one_col_elem_num, \
                                   out_dptr);                              \
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

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_RAW_DECODER, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
