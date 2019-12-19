#include "oneflow/core/record/ofrecord_bytes_list_encoder.h"

namespace oneflow {

template<typename T>
void OFRecordEncoderImpl<EncodeCase::kBytesList, T>::EncodeOneCol(
    DeviceCtx* ctx, const Blob* in_blob, int64_t in_offset, Feature& feature,
    const std::string& field_name, int64_t one_col_elem_num) const {
  UNIMPLEMENTED();
}

#define INSTANTIATE_OFRECORD_BYTES_LIST_ENCODER(type_cpp, type_proto) \
  template class OFRecordEncoderImpl<EncodeCase::kBytesList, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_BYTES_LIST_ENCODER,
                     ((char, DataType::kChar))((int8_t, DataType::kInt8)))

}  // namespace oneflow
