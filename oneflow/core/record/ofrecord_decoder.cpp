#include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/record/ofrecord_raw_decoder.h"

namespace oneflow {

template<EncodeType encode_type, typename T>
int32_t OFRecordDecoder<encode_type, T>::DecodeOneCol(
    DeviceCtx* ctx, RecordBlob<OFRecord>* record_blob, const std::string& name,
    int32_t col_id, Blob* out_blob) const {
  int32_t max_col_id = 0;
  if (out_blob->has_col_num_field()) {
    max_col_id = ReadColNum(ctx, record_blob, name, out_blob) - 1;
  }
  if (out_blob->has_data_id_field()) { ReadDataId(ctx, record_blob, out_blob); }
  ReadDataContent(ctx, record_blob, name, col_id, out_blob);
  return max_col_id;
}

template<EncodeType encode_type, typename T>
int32_t OFRecordDecoder<encode_type, T>::ReadColNum(
    DeviceCtx* ctx, RecordBlob<OFRecord>* record_blob, const std::string& name,
    Blob* out_blob) const {
  int32_t i = 0;
  int32_t max_col_num = 0;
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at(name);
    int32_t col_num = GetColNumOfFeature(feature, out_blob->shape().Count(1));
    max_col_num = std::max(max_col_num, col_num);
    out_blob->set_col_num(i++, col_num);
  });
  CHECK_GT(max_col_num, 0);
  while (i < out_blob->shape().At(0)) { out_blob->set_col_num(i++, 0); }
  return max_col_num;
}

template<EncodeType encode_type, typename T>
void OFRecordDecoder<encode_type, T>::ReadDataId(
    DeviceCtx* ctx, RecordBlob<OFRecord>* record_blob, Blob* out_blob) const {
  int64_t max_data_id_size = JobDesc::Singleton()->SizeOfOneDataId();
  int32_t i = 0;
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at("data_id");
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    const std::string& data_id_str = feature.bytes_list().value(0);
    CHECK_LE(data_id_str.size(), max_data_id_size);
    Memcpy<DeviceType::kCPU>(ctx, out_blob->mut_data_id(i), data_id_str.c_str(),
                             data_id_str.size());
    if (data_id_str.size() != max_data_id_size) {
      *(out_blob->mut_data_id(i) + data_id_str.size()) = '\0';
    }
    i += 1;
  });
  int64_t left_row_num = out_blob->shape().At(0) - i;
  if (left_row_num > 0) {
    Memset<DeviceType::kCPU>(ctx, out_blob->mut_data_id(i), '\0',
                             left_row_num * max_data_id_size);
  }
}

template<EncodeType encode_type, typename T>
void OFRecordDecoder<encode_type, T>::ReadDataContent(
    DeviceCtx* ctx, RecordBlob<OFRecord>* record_blob, const std::string& name,
    int32_t col_id, Blob* out_blob) const {
  int64_t one_col_elem_num = out_blob->shape().Count(1);
  int32_t i = 0;
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at(name);
    T* out_dptr = out_blob->mut_dptr<T>() + i * one_col_elem_num;
    if (col_id < out_blob->col_num(i)) {
      ReadOneCol(ctx, feature, col_id, out_dptr, one_col_elem_num);
    } else {
      Memset<DeviceType::kCPU>(ctx, out_dptr, 0, one_col_elem_num * sizeof(T));
    }
    i += 1;
  });
  int64_t left_row_num = out_blob->shape().At(0) - i;
  if (left_row_num > 0) {
    Memset<DeviceType::kCPU>(ctx,
                             out_blob->mut_dptr<T>() + i * one_col_elem_num, 0,
                             left_row_num * one_col_elem_num * sizeof(T));
  }
}

OFRecordDecoderIf* GetOFRecordDecoder(EncodeType encode_type,
                                      DataType data_type) {
  static const HashMap<std::string, OFRecordDecoderIf*> obj = {

#define MAKE_ENTRY(et, dt)                \
  {GetHashKey(et, OF_PP_PAIR_SECOND(dt)), \
   new OFRecordDecoderImpl<et, OF_PP_PAIR_FIRST(dt)>},

      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ENCODE_TYPE_SEQ,
                                       ARITHMETIC_DATA_TYPE_SEQ)

  };
  return obj.at(GetHashKey(encode_type, data_type));
}

}  // namespace oneflow
