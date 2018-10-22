#include "oneflow/core/record/ofrecord_raw_encoder.h"

namespace oneflow {

namespace {

template<typename T>
void CopyToFeature(Feature& feature, const std::string& field_name, const T* in_dptr,
                   size_t elem_num) {
  DataType data_type = GetDataType<T>();
  if (data_type == DataType::kInt8) {
    feature.mutable_bytes_list()->add_value(reinterpret_cast<const char*>(in_dptr), elem_num);
  }
#define DEFINE_ONE_ELIF(CppT, ListT)                                                    \
  else if (data_type == GetDataType<CppT>()) {                                          \
    feature.mutable_##ListT##_list()->mutable_value()->Resize(elem_num, 0);             \
    CppT* out_dptr = feature.mutable_##ListT##_list()->mutable_value()->mutable_data(); \
    Memcpy<DeviceType::kCPU>(nullptr, out_dptr, in_dptr, elem_num * sizeof(T));         \
  }
  DEFINE_ONE_ELIF(float, float)
  DEFINE_ONE_ELIF(double, double)
  DEFINE_ONE_ELIF(int32_t, int32)
#undef DEFINE_ONE_ELIF
  else {
    UNIMPLEMENTED();
  }
}

}  // namespace

template<typename T>
void OFRecordEncoderImpl<EncodeCase::kRaw, T>::EncodeOneCol(DeviceCtx* ctx, const Blob* in_blob,
                                                            int64_t in_offset, Feature& feature,
                                                            const std::string& field_name,
                                                            int64_t one_col_elem_num) const {
  const auto& shape = in_blob->shape();
  CHECK(!in_blob->has_dim2_valid_num_field());
  CHECK_EQ(one_col_elem_num, shape.Count(1));
  CHECK_EQ(in_offset % one_col_elem_num, 0);
  int64_t dim0_idx = in_offset / one_col_elem_num;
  const T* in_dptr = in_blob->dptr<T>() + in_offset;
  int64_t elem_num = shape.NumAxes() == 1 ? 1 : in_blob->dim1_valid_num(dim0_idx) * shape.Count(2);
  CopyToFeature(feature, field_name, in_dptr, elem_num);
}

template<typename T>
void OFRecordEncoderImpl<EncodeCase::kRaw, T>::EncodeMultiCol(
    DeviceCtx*, const Blob* in_blob, const std::vector<int64_t>& in_offsets, Feature& feature,
    const std::string& field_name, int64_t one_col_elem_num) const {
  CHECK(!in_blob->has_dim1_valid_num_field());
  CHECK(!in_blob->has_dim2_valid_num_field());
  CHECK_EQ(one_col_elem_num, in_blob->shape().Count(1));
  size_t elem_num = in_offsets.size() * one_col_elem_num;
  std::unique_ptr<T[]> buf(new T[elem_num]);
  FOR_RANGE(int32_t, i, 0, in_offsets.size()) {
    const T* cur_in_dptr = in_blob->dptr<T>() + in_offsets.at(i);
    T* cur_out_dptr = &buf[i * one_col_elem_num];
    Memcpy<DeviceType::kCPU>(nullptr, cur_out_dptr, cur_in_dptr, one_col_elem_num * sizeof(T));
  }
  CopyToFeature(feature, field_name, &buf[0], elem_num);
}

#define INSTANTIATE_OFRECORD_RAW_ENCODER(type_cpp, type_proto) \
  template class OFRecordEncoderImpl<EncodeCase::kRaw, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_RAW_ENCODER, ARITHMETIC_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ)

}  // namespace oneflow
