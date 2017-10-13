#include "oneflow/core/kernel/data_set_util.h"
#include "opencv2/opencv.hpp"

namespace oneflow {

std::unique_ptr<DataSetHeader> DataSetUtil::CreateHeader(
    const std::string& type, uint32_t data_item_count,
    const std::vector<uint32_t>& dim_array) {
  std::unique_ptr<DataSetHeader> header(new DataSetHeader);
  header->data_item_count = data_item_count;
  CHECK(type.size() <= sizeof(header->type));
  type.copy(header->type, type.size(), 0);
  CHECK(dim_array.size() <= sizeof(header->dim_array));
  header->dim_array_size = dim_array.size();
  for (int i = 0; i < dim_array.size(); ++i) {
    header->dim_array[i] = dim_array[i];
  }
  return std::move(header);
}

std::unique_ptr<DataSetLabelDesc, decltype(&free)> DataSetUtil::CreateLabelDesc(
    const std::vector<std::string>& labels) {
  auto header = Malloc<DataSetLabelDesc>(labels.size());
  header->label_array_size = labels.size();
  for (int i = 0; i < labels.size(); i++) {
    const auto& label = labels.at(i);
    CHECK(label.size() < sizeof(header->label_desc[0]));
    memset(header->label_desc[i], 0, sizeof(header->label_desc[0]));
    label.copy(header->label_desc[i], label.size(), 0);
  }
  return header;
}

std::unique_ptr<DataItem, decltype(&free)> DataSetUtil::CreateDataItem(
    const DataSetHeader& header) {
  size_t elem_cnt = header.TensorElemCount();
  auto body = Malloc<DataItem>(elem_cnt);
  body->len = elem_cnt;
  return std::move(body);
}

std::unique_ptr<DataItem, decltype(&free)> DataSetUtil::CreateLabelItem(
    const DataSetHeader& header, const std::vector<uint32_t>& label_index) {
  auto data_item = CreateDataItem(header);
  CHECK(label_index.size() <= data_item->len);
  for (int i = 0; i < label_index.size(); ++i) {
    data_item->data[i] = label_index[i];
  }
  return std::move(data_item);
}

std::unique_ptr<DataItem, decltype(&free)> DataSetUtil::CreateImageItem(
    const DataSetHeader& header, const std::string& img_file_path) {
  uint32_t width, height;
  width = header.dim_array[1];
  height = header.dim_array[2];
  cv::Mat img = cv::imread(img_file_path);
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(width, height));
  auto getter = [&](uint32_t d, uint32_t row, uint32_t col) -> double {
    return resized.at<cv::Vec3b>(row, col)[d];
  };
  auto data_item = CreateDataItem(header);
  LoadImageData(data_item.get(), width, height, getter);
  return std::move(data_item);
}

void DataSetUtil::LoadImageData(
    DataItem* body, uint32_t width, uint32_t height,
    const std::function<double(uint32_t d, uint32_t r, uint32_t c)>& Get) {
  uint32_t image_size = width * height;
  for (uint32_t i = 0; i < body->len; i++) {
    body->data[i] = Get(i / image_size, (i / width) % height, i % width);
  }
}

}  // namespace oneflow
