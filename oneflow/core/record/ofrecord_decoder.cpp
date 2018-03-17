#include "oneflow/core/record/ofrecord_decoder.h"

namespace oneflow {

template<EncodeType E, DataType D>
int32_t OFRecordDecoder<E, D>::Decode(RecordBlob<OFRecord>* record_blob,
                                      const std::string& name, int32_t col_id,
                                      Blob* out_blob, DeviceCtx* ctx) {
  int32_t max_col_id = 0;
  if (out_blob->has_col_num_field()) {
    max_col_id = ReadColNumToOutBlob(record_blob, name, out_blob) - 1;
  }
  if (out_blob->has_data_id_field()) {
    ReadDataIdToOutBlob(record_blob, out_blob, ctx);
  }
  ReadDataContentToOutBlob(record_blob, name, col_id, out_blob, ctx);
  return max_col_id;
}

template<EncodeType E, DataType D>
int32_t OFRecordDecoder<E, D>::ReadColNumToOutBlob(
    RecordBlob<OFRecord>* record_blob, const std::string& name,
    Blob* out_blob) {
  CHECK(out_blob->has_col_num_field());
  int32_t i = 0;
  int32_t max_col_num = 1;
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at(name);
    int32_t col_num = GetColNumOfFeature(feature, out_blob->shape().Count(1));
    CHECK(col_num <= out_blob->max_col_num());
    max_col_num = std::max(max_col_num, col_num);
    out_blob->set_col_num(i++, col_num);
  });
  while (i < JobDesc::Singleton()->SinglePieceSize()) {
    out_blob->set_col_num(i++, 0);
  }
  return max_col_num;
}

template<EncodeType E, DataType D>
void OFRecordDecoder<E, D>::ReadDataIdToOutBlob(
    RecordBlob<OFRecord>* record_blob, Blob* out_blob, DeviceCtx* ctx) {
  CHECK(out_blob->has_data_id_field());
  int32_t i = 0;
  size_t size_of_data_id = JobDesc::Singleton()->SizeOfOneDataId();
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at("data_id");
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    const std::string& data_id_str = feature.bytes_list().value(0);
    CHECK_LE(data_id_str.size(), size_of_data_id);
    Memcpy<DeviceType::kCPU>(ctx, out_blob->mut_data_id(i++), &data_id_str,
                             data_id_str.size());
  });
  memset(out_blob->mut_data_id(i), 0,
         size_of_data_id * (JobDesc::Singleton()->SinglePieceSize() - i));
}

template<EncodeType E, DataType D>
void OFRecordDecoder<E, D>::ReadDataContentToOutBlob(
    RecordBlob<OFRecord>* record_blob, const std::string& name, int32_t col_id,
    Blob* out_blob, DeviceCtx* ctx) {
  int32_t i = 0;
  int64_t item_size =
      out_blob->shape().Count(1) * GetSizeOfDataType(out_blob->data_type());
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at(name);
    int32_t col_num = out_blob->has_col_num_field() ? out_blob->col_num(i) : 1;
    if (col_id < col_num) {
      ReadDataContentForOneItem(feature, col_id, out_blob, ctx);
    } else {
      memset(out_blob->mut_dptr<char>() + i * item_size, 0, item_size);
    }
    ++i;
  });
  memset(out_blob->mut_dptr<char>() + i * item_size, 0,
         item_size * (JobDesc::Singleton()->SinglePieceSize() - i));
}

DataType DataTypeOf(const Feature& feature) {
  if (feature.has_bytes_list()) {
    return DataType::kInt8;
  } else if (feature.has_float_list()) {
    return DataType::kFloat;
  } else if (feature.has_double_list()) {
    return DataType::kDouble;
  } else if (feature.has_int32_list()) {
    return DataType::kInt32;
  } else {
    UNIMPLEMENTED();
  }
}

int64_t SizeOf(const Feature& feature) {
  if (feature.has_bytes_list()) {
    return feature.bytes_list().value_size();
  } else if (feature.has_float_list()) {
    return feature.float_list().value_size();
  } else if (feature.has_double_list()) {
    return feature.double_list().value_size();
  } else if (feature.has_int32_list()) {
    return feature.int32_list().value_size();
  } else {
    UNIMPLEMENTED();
  }
}

template class OFRecordDecoder<EncodeType::kRaw, DataType::kInt8>;
template class OFRecordDecoder<EncodeType::kRaw, DataType::kInt32>;
template class OFRecordDecoder<EncodeType::kRaw, DataType::kFloat>;
template class OFRecordDecoder<EncodeType::kRaw, DataType::kDouble>;
template class OFRecordDecoder<EncodeType::kJpeg, DataType::kInt8>;
template class OFRecordDecoder<EncodeType::kJpeg, DataType::kInt32>;
template class OFRecordDecoder<EncodeType::kJpeg, DataType::kFloat>;
template class OFRecordDecoder<EncodeType::kJpeg, DataType::kDouble>;

}  // namespace oneflow
