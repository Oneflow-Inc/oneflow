#include "oneflow/core/persistence/data_set_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "opencv2/opencv.hpp"

namespace oneflow {

uint8_t DataSetUtil::ValidateRecordMeta(const Record& buffer) {
  uint8_t check_sum = 0;
  int meta_len = FlexibleSizeOf<Record>(0);
  for (int i = 0; i < meta_len; ++i) {
    check_sum += reinterpret_cast<const char*>(&buffer)[i];
  }
  return check_sum;
}

uint8_t DataSetUtil::ValidateRecord(const Record& buffer) {
  return ValidateRecordMeta(buffer);
}

std::unique_ptr<Record, decltype(&free)> DataSetUtil::NewRecord(
    const std::string& key, size_t value_buf_len, DataType dtype,
    DataCompressType dctype, const std::function<void(char* buff)>& Fill) {
  size_t value_offset = RoundUp(key.size(), 8);
  auto buffer = FlexibleMalloc<Record>(value_buf_len + value_offset);
  buffer->data_type = dtype;
  buffer->data_compress_type = dctype;
  buffer->key_len = key.size();
  buffer->value_offset = value_offset;
  memset(buffer->data, 0, value_offset);
  key.copy(buffer->mut_key_buffer(), key.size());
  if (value_buf_len) { Fill(const_cast<char*>(buffer->mut_value_buffer())); }
  UpdateRecordCheckSum(buffer.get());
  return buffer;
}

void DataSetUtil::UpdateRecordMetaCheckSum(Record* buffer) {
  uint8_t meta_check_sum = 0;
  const int meta_len = FlexibleSizeOf<Record>(0);
  for (int i = 0; i < meta_len; ++i) {
    meta_check_sum += reinterpret_cast<char*>(buffer)[i];
  }
  meta_check_sum -= buffer->meta_check_sum;
  buffer->meta_check_sum = -meta_check_sum;
}

void DataSetUtil::UpdateRecordCheckSum(Record* buffer) {
  uint8_t data_check_sum = 0;
  for (int i = 0; i < buffer->len; ++i) { data_check_sum += buffer->data[i]; }
  buffer->data_check_sum = -data_check_sum;
  UpdateRecordMetaCheckSum(buffer);
}

std::unique_ptr<DataSetHeader> DataSetUtil::CreateHeader(
    const std::string& type, uint32_t data_item_count,
    const std::vector<uint32_t>& dim_array) {
  std::unique_ptr<DataSetHeader> header(new DataSetHeader);
  CHECK(type.size() <= sizeof(header->type));
  type.copy(header->type, type.size(), 0);
  header->data_item_count = data_item_count;
  CHECK(dim_array.size() <= sizeof(header->dim_array));
  header->dim_array_size = dim_array.size();
  memset(header->dim_array, 0, sizeof(header->dim_array));
  for (int i = 0; i < dim_array.size(); ++i) {
    header->dim_array[i] = dim_array[i];
  }
  UpdateHeaderCheckSum(header.get());
  CHECK(!ValidateHeader(*header));
  return header;
}

uint32_t DataSetUtil::ValidateHeader(const DataSetHeader& header) {
  static_assert(!(sizeof(DataSetHeader) % sizeof(uint32_t)), "no alignment");
  uint32_t check_sum = 0;
  int len = sizeof(DataSetHeader) / sizeof(uint32_t);
  for (int i = 0; i < len; ++i) {
    check_sum += reinterpret_cast<const uint32_t*>(&header)[i];
  }
  return check_sum;
}

void DataSetUtil::UpdateHeaderCheckSum(DataSetHeader* header) {
  static_assert(!(sizeof(DataSetHeader) % sizeof(uint32_t)), "no alignment");
  uint32_t check_sum = 0;
  int len = sizeof(DataSetHeader) / sizeof(uint32_t);
  for (int i = 0; i < len; ++i) {
    check_sum += reinterpret_cast<uint32_t*>(header)[i];
  }
  check_sum -= header->check_sum;
  header->check_sum = -check_sum;
}

std::unique_ptr<Record, decltype(&free)> DataSetUtil::CreateLabelItem(
    const DataSetHeader& header, const std::string& key, uint32_t label_index) {
  auto buffer = NewRecord(
      key, sizeof(uint32_t), DataType::kUInt32,
      [=](char* data) { *reinterpret_cast<uint32_t*>(data) = label_index; });
  return buffer;
}

std::unique_ptr<Record, decltype(&free)> DataSetUtil::CreateImageItem(
    const DataSetHeader& header, const std::string& img_file_path) {
  cv::Mat img = cv::imread(img_file_path);
  std::vector<unsigned char> raw_buf;
  std::vector<int> param{CV_IMWRITE_JPEG_QUALITY, 95};
  cv::imencode(".jpg", img, raw_buf, param);
  auto buffer = NewRecord(
      img_file_path, raw_buf.size(), DataType::kChar, DataCompressType::kJpeg,
      [&](char* data) { memcpy(data, raw_buf.data(), raw_buf.size()); });
  return buffer;
}

}  // namespace oneflow
