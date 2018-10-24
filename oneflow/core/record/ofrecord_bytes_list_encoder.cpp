#include "oneflow/core/record/ofrecord_bytes_list_encoder.h"

namespace oneflow {

template<typename T>
void OFRecordEncoderImpl<EncodeCase::kBytesList, T>::EncodeOneCol(
    DeviceCtx* ctx, const Blob* in_blob, int64_t in_offset, Feature& feature,
    const std::string& field_name, int64_t one_col_elem_num) const {
  static_assert(sizeof(T) == 1, "only char and int8_t supported");
  CHECK_GE(in_blob->shape().NumAxes(), 3);
  int64_t dim0_elem_num = in_blob->shape().Count(1);
  CHECK_EQ(one_col_elem_num, dim0_elem_num);
  CHECK_EQ(in_offset % dim0_elem_num, 0);
  int64_t dim1_elem_num = in_blob->shape().Count(2);
  int64_t dim2_elem_num = in_blob->shape().Count(3);
  int64_t dim0_idx = in_offset / dim0_elem_num;

  // dim1_valid_num maybe 0
  feature.mutable_bytes_list();
  FOR_RANGE(int64_t, dim1_idx, 0, in_blob->dim1_valid_num(dim0_idx)) {
    int64_t bytes_size = in_blob->dim2_valid_num(dim0_idx, dim1_idx) * dim2_elem_num;
    feature.mutable_bytes_list()->add_value(
        in_blob->dptr<char>() + in_offset + dim1_idx * dim1_elem_num, bytes_size);
  }
}

template<typename T>
void OFRecordEncoderImpl<EncodeCase::kBytesList, T>::EncodeMultiCol(
    DeviceCtx*, const Blob* in_blob, const std::vector<int64_t>& in_offsets, Feature& feature,
    const std::string& field_name, int64_t one_col_elem_num) const {
  static_assert(sizeof(T) == 1, "only char and int8_t supported");
  CHECK_GE(in_blob->shape().NumAxes(), 2);
  int64_t dim0_elem_num = in_blob->shape().Count(1);
  CHECK_EQ(one_col_elem_num, dim0_elem_num);
  int64_t dim1_elem_num = in_blob->shape().Count(2);

  CHECK(!in_blob->has_dim2_valid_num_field());
  CHECK_GT(in_offsets.size(), 0);
  for (int64_t in_offset : in_offsets) {
    CHECK_EQ(in_offset % dim0_elem_num, 0);
    int64_t dim0_idx = in_offset / dim0_elem_num;
    feature.mutable_bytes_list()->add_value(in_blob->dptr<char>() + in_offset,
                                            in_blob->dim1_valid_num(dim0_idx) * dim1_elem_num);
  }
}

#define INSTANTIATE_OFRECORD_BYTES_LIST_ENCODER(type_cpp, type_proto) \
  template class OFRecordEncoderImpl<EncodeCase::kBytesList, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_BYTES_LIST_ENCODER,
                     ((char, DataType::kChar))((int8_t, DataType::kInt8)))

}  // namespace oneflow
