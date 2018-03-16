#include "oneflow/core/record/record_decoder.h"

namespace oneflow {

template<typename T>
int32_t RecordDecoder<T>::ReadRecordToOutBlob(RecordBlob<OFRecord>* record_blob,
                                              const std::string& name,
                                              int32_t cur_col_id,
                                              Blob* out_blob, DeviceCtx* ctx){
  int32_t max_col_id = 0;
  if (out_blob->has_col_num_field()) {
    max_col_id = ReadColNumToOutBlob();
  }
  if (out_blob->has_data_id_field()) {
    ReadDataIdToOutBlob();
  }
  ReadDataContentToOutBlob();
}

template<typename T>
int32_t RecordDecoder<T>::ReadColNumToOutBlob(){
  CHECK(out_blob->has_col_num_field());
  int32_t i = 0;
  int32_t max_col_id = 0;
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at(name);
    int32_t col_num = // TODO ;
    CHECK(col_num <= out_blob->max_col_num());
    max_col_id = std::max(max_col_id, col_num - 1);
    out_blob->set_col_num(i++, col_num);
  });
  while (i < JobDesc::Singleton()->SinglePieceSize()) {
    out_blob->set_col_num(i++, 0);
  }
}

template<typename T>
void RecordDecoder<T>::ReadDataIdToOutBlob() {
  CHECK(out_blob->has_data_id_field());
  int32_t i = 0;
  size_t size_of_data_id = JobDesc::Singleton()->SizeOfOneDataId();
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at("data_id");
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    const std::string& data_id_str = feature.bytes_list().value(0);
    CHECK(data_id_str.size() <= size_of_data_id);
    Memcpy<DeviceType::kCPU>(device_ctx, out_blob->mut_data_id(i++),
                             &data_id_str, data_id_str.size());
  });
  memset(out_blob->mut_data_id(i), 0,
         size_of_data_id * (JobDesc::Singleton()->SinglePieceSize() - i));
}
  
template<typename T>
void RecordDecoder<T>::ReadDataContentToOutBlob() {
  int32_t i = 0;
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at(name);
    // get pointer
    const void* dptr = info.first;
    int32_t col_num = info.second;
    if (col_id < col_num) {
      // read one row
      if (encode_type == kRaw) {
        dptr = dptr + col_id * col_size;
      } else if (encode_type == kJpeg) {
        dptr = *(dptr + col_id);
      } else {
        UNIMPLEMENTED();
      }
      Memcpy<DeviceType::kCPU>(device_ctx, out_blob->mut_dptr() + col_size * i,
                               dptr, col_size);
    } else {
      // set one row to zero
      memset(out_blob->mut_dptr() + col_size * i, 0, col_size);
    }
    ++i;
  });
  memset(out_blob->mut_dptr() + col_size * i, 0,
         col_size * (JobDesc::Singleton()->SinglePieceSize() - i));

}

}  // namespace oneflow
