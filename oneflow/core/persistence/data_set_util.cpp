#include "oneflow/core/persistence/data_set_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "opencv2/opencv.hpp"

namespace oneflow {

uint8_t DataSetUtil::ValidateBufferMeta(const Buffer& buffer) {
  uint8_t check_sum;
  int meta_len = FlexibleSizeOf<Buffer>(0);
  for (int i = 0; i < meta_len; ++i) {
    check_sum += reinterpret_cast<const char*>(&buffer)[i];
  }
  return check_sum;
}

uint8_t DataSetUtil::ValidateBuffer(const Buffer& buffer) {
  return ValidateBufferMeta(buffer);
}

std::unique_ptr<Buffer, decltype(&free)> DataSetUtil::NewBuffer(
    size_t len, DataType dtype, DataCompressType dctype,
    const std::function<void(char* buff)>& Fill) {
  auto buffer = FlexibleMalloc<Buffer>(len);
  buffer->data_type = dtype;
  buffer->data_compress_type = dctype;
  Fill(buffer->data);
  UpdateBufferCheckSum(buffer.get());
  return buffer;
}

void DataSetUtil::UpdateBufferMetaCheckSum(Buffer* buffer) {
  uint8_t meta_check_sum = 0;
  const int meta_len = FlexibleSizeOf<Buffer>(0);
  for (int i = 0; i < meta_len; ++i) {
    meta_check_sum += reinterpret_cast<char*>(buffer)[i];
  }
  meta_check_sum -= buffer->meta_check_sum;
  buffer->meta_check_sum = -meta_check_sum;
}

void DataSetUtil::UpdateBufferCheckSum(Buffer* buffer) {
  uint8_t data_check_sum = 0;
  for (int i = 0; i < buffer->len; ++i) { data_check_sum += buffer->data[i]; }
  buffer->data_check_sum = -data_check_sum;
  UpdateBufferMetaCheckSum(buffer);
}

std::unique_ptr<DataSetHeader> DataSetUtil::CreateHeader(
    const std::string& type, DataType dtype, uint32_t data_item_count,
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
  return std::move(header);
}

std::unique_ptr<Buffer, decltype(&free)> DataSetUtil::CreateLabelItem(
    const DataSetHeader& header, uint32_t label_index) {
  auto buffer = NewBuffer(sizeof(uint32_t), DataType::kUInt32, [=](char* data) {
    *reinterpret_cast<uint32_t*>(data) = label_index;
  });
  return buffer;
}

std::unique_ptr<Buffer, decltype(&free)> DataSetUtil::CreateImageItem(
    const DataSetHeader& header, const std::string& img_file_path) {
  cv::Mat img = cv::imread(img_file_path);
  std::vector<unsigned char> raw_buf;
  std::vector<int> param{CV_IMWRITE_JPEG_QUALITY, 95};
  cv::imencode(".jpg", img, raw_buf, param);
  auto buffer = NewBuffer(
      raw_buf.size(), DataType::kChar, DataCompressType::kJpeg,
      [&](char* data) { memcpy(data, raw_buf.data(), raw_buf.size()); });
  return buffer;
}

}  // namespace oneflow
