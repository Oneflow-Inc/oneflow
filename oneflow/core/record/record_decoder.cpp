#include "oneflow/core/record/record_decoder.h"

namespace oneflow {

template<typename T>
int32_t RecordDecoder<T>::ReadRecordToOutBlob(RecordBlob<OFRecord>* record_blob,
                                              const std::string& name,
                                              int32_t cur_col_id,
                                              Blob* out_blob, DeviceCtx* ctx) {
  int32_t max_col_id = 0;
  if (out_blob->has_col_num_field()) {
    max_col_id = ReadColNumToOutBlob(record_blob, name, out_blob);
  }
  if (out_blob->has_data_id_field()) {
    ReadDataIdToOutBlob(record_blob, out_blob, ctx);
  }
  ReadDataContentToOutBlob(record_blob, name, cur_col_id, out_blob, ctx);
  return max_col_id;
}

template<typename T>
int32_t RecordDecoder<T>::ReadColNumToOutBlob(RecordBlob<OFRecord>* record_blob,
                                              const std::string& name,
                                              Blob* out_blob) {
  CHECK(out_blob->has_col_num_field());
  int32_t i = 0;
  int32_t max_col_id = 0;
  record_blob->ForEachRecord([&](const OFRecord& record) {
    Feature& feature = record.feature().at(name);
    int32_t col_num = GetColNumOfFeature(feature, out_blob->shape().Count(1));
    CHECK(col_num <= out_blob->max_col_num());
    max_col_id = std::max(max_col_id, col_num - 1);
    out_blob->set_col_num(i++, col_num);
  });
  while (i < JobDesc::Singleton()->SinglePieceSize()) {
    out_blob->set_col_num(i++, 0);
  }
  return max_col_id;
}

template<typename T>
void RecordDecoder<T>::ReadDataIdToOutBlob(RecordBlob<OFRecord>* record_blob,
                                           Blob* out_blob, DeviceCtx* ctx) {
  CHECK(out_blob->has_data_id_field());
  int32_t i = 0;
  size_t size_of_data_id = JobDesc::Singleton()->SizeOfOneDataId();
  record_blob->ForEachRecord([&](const OFRecord& record) {
    Feature& feature = record.feature().at("data_id");
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    const std::string& data_id_str = feature.bytes_list().value(0);
    CHECK(data_id_str.size() <= size_of_data_id);
    Memcpy<DeviceType::kCPU>(ctx, out_blob->mut_data_id(i++), &data_id_str,
                             data_id_str.size());
  });
  memset(out_blob->mut_data_id(i), 0,
         size_of_data_id * (JobDesc::Singleton()->SinglePieceSize() - i));
}

template<typename T>
void RecordDecoder<T>::ReadDataContentToOutBlob(
    RecordBlob<OFRecord>* record_blob, const std::string& name,
    int32_t cur_col_id, Blob* out_blob, DeviceCtx* ctx) {
  int32_t i = 0;
  int64_t item_size = out_blob->shape().Count(1);
  T* out_dptr = out_blob->mut_dptr<T>();
  record_blob->ForEachRecord([&](const OFRecord& record) {
    Feature& feature = record.feature().at(name);
    int32_t col_num = out_blob->has_col_num_field() ? out_blob->col_num(i) : 1;
    if (cur_col_id < col_num) {
      ReadDataContentForOneItem(out_dptr, feature, item_size, ctx);
    } else {
      memset(out_dptr, 0, item_size);
    }
    ++i;
    out_dptr += item_size;
  });
  memset(out_dptr, 0,
         item_size * (JobDesc::Singleton()->SinglePieceSize() - i));
}

}  // namespace oneflow
