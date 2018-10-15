#include "oneflow/core/record/ofrecord_bytes_list_decoder.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kBytesList, T>::SetDim1ValidNum(const Feature& feature,
                                                                     Blob* out_blob,
                                                                     int32_t dim0_idx) const {
  static_assert(sizeof(T) == 1, "only char and int8_t supported");
  CHECK_GE(out_blob->static_shape().NumAxes(), 2);
  CHECK(feature.has_bytes_list());
  int64_t dim1_valid_num = feature.bytes_list().value_size();
  CHECK_LE(dim1_valid_num, out_blob->static_shape().At(1));
  out_blob->set_dim1_valid_num(dim0_idx, dim1_valid_num);
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kBytesList, T>::SetDim2ValidNum(const Feature& feature,
                                                                     Blob* out_blob,
                                                                     int32_t dim0_idx) const {
  static_assert(sizeof(T) == 1, "only char and int8_t supported");
  CHECK_GE(out_blob->static_shape().NumAxes(), 3);
  CHECK(feature.has_bytes_list());
  FOR_RANGE(int32_t, dim1_idx, 0, feature.bytes_list().value_size()) {
    out_blob->set_dim2_valid_num(dim0_idx, dim1_idx, feature.bytes_list().value(dim1_idx).size());
  }
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
    CopyElem(in_dptr, out_dptr, shape_count2);
    out_dptr += shape_count2;
  }
}

#define INSTANTIATE_OFRECORD_BYTES_LIST_DECODER(type_cpp, type_proto) \
  template class OFRecordDecoderImpl<EncodeCase::kBytesList, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_BYTES_LIST_DECODER,
                     ((char, DataType::kChar))((int8_t, DataType::kInt8)))

}  // namespace oneflow
