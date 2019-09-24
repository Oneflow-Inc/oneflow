#include "oneflow/core/record/ofrecord_bytes_list_decoder.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kBytesList, T>::SetDim1ValidNum(const Feature& feature,
                                                                     Blob* out_blob,
                                                                     int64_t dim0_idx) const {
  static_assert(sizeof(T) == 1, "only char and int8_t supported");
  UNIMPLEMENTED();
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kBytesList, T>::SetDim2ValidNum(const Feature& feature,
                                                                     Blob* out_blob,
                                                                     int64_t dim0_idx) const {
  static_assert(sizeof(T) == 1, "only char and int8_t supported");
  UNIMPLEMENTED();
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kBytesList, T>::ReadOneCol(
    DeviceCtx* ctx, const Feature& feature, const BlobConf& blob_conf, int32_t col_id, T* out_dptr,
    int64_t one_col_elem_num, std::function<int32_t(void)> NextRandomInt) const {
  static_assert(sizeof(T) == 1, "only char and int8_t supported");
  CHECK(feature.has_bytes_list());
  int64_t shape_count2 = Shape(blob_conf.shape()).Count(1);  // yes, Count(1) is right
  for (const auto& bytes : feature.bytes_list().value()) {
    CHECK_LE(bytes.size(), shape_count2);
    auto* in_dptr = reinterpret_cast<const T*>(bytes.c_str());
    CopyElem(in_dptr, out_dptr, bytes.size());
    out_dptr += shape_count2;
  }
}

#define INSTANTIATE_OFRECORD_BYTES_LIST_DECODER(type_cpp, type_proto) \
  template class OFRecordDecoderImpl<EncodeCase::kBytesList, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_BYTES_LIST_DECODER,
                     ((char, DataType::kChar))((int8_t, DataType::kInt8)))

}  // namespace oneflow
