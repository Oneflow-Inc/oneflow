#include "oneflow/core/record/ofrecord_raw_encoder.h"

namespace oneflow {

template<typename T>
void OFRecordEncoderImpl<EncodeCase::kRaw, T>::EncodeOneCol(
    DeviceCtx* ctx, const Blob* in_blob, int64_t in_offset, Feature& feature,
    const std::string& field_name, int64_t one_col_elem_num) const {
  const T* in_dptr = in_blob->dptr<T>() + in_offset;
  DataType data_type = GetDataType<T>();
  if (data_type == DataType::kInt8) {
    feature.mutable_bytes_list()->add_value(
        reinterpret_cast<const char*>(in_dptr), one_col_elem_num);
  }
#define DEFINE_ONE_ELIF(CppT, ListT)                                       \
  else if (data_type == GetDataType<CppT>()) {                             \
    feature.mutable_##ListT##_list()->mutable_value()->Resize(             \
        one_col_elem_num, 0);                                              \
    CppT* out_dptr =                                                       \
        feature.mutable_##ListT##_list()->mutable_value()->mutable_data(); \
    Memcpy<DeviceType::kCPU>(nullptr, out_dptr, in_dptr,                   \
                             one_col_elem_num * sizeof(T));                \
  }
  DEFINE_ONE_ELIF(float, float)
  DEFINE_ONE_ELIF(double, double)
  DEFINE_ONE_ELIF(int32_t, int32)
#undef DEFINE_ONE_ELIF
  else {
    UNIMPLEMENTED();
  }
}

#define INSTANTIATE_OFRECORD_RAW_ENCODER(type_cpp, type_proto) \
  template class OFRecordEncoderImpl<EncodeCase::kRaw, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_RAW_ENCODER, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
