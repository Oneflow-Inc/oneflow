#include "oneflow/core/kernel/data_set_util.h"
#include "opencv2/opencv.hpp"

namespace oneflow {

std::unique_ptr<DataSetHeaderDesc> DataSetUtil::CreateHeaderDesc(
    const DataSetLabelHeader& header, uint32_t data_item_size) {
  std::unique_ptr<DataSetHeaderDesc> desc(new DataSetHeaderDesc);
  strcpy(desc->type, "label");
  desc->data_item_size = data_item_size;
  desc->header_buffer_len = FlexibleSizeOf(header);
  return std::move(desc);
}

std::unique_ptr<DataSetHeaderDesc> DataSetUtil::CreateHeaderDesc(
    const DataSetFeatureHeader& header, uint32_t data_item_size) {
  std::unique_ptr<DataSetHeaderDesc> desc(new DataSetHeaderDesc);
  strcpy(desc->type, "feature");
  desc->data_item_size = data_item_size;
  desc->header_buffer_len = FlexibleSizeOf(header);
  return std::move(desc);
}

std::unique_ptr<DataSetFeatureHeader, decltype(&free)>
DataSetUtil::CreateImageFeatureHeader(uint32_t width, uint32_t height) {
  auto header = Malloc<DataSetFeatureHeader>(3);
  header->dim_array_size = 3;
  header->dim_vec[0] = 3;
  header->dim_vec[1] = width;
  header->dim_vec[2] = height;
  return std::move(header);
}

std::unique_ptr<DataSetLabelHeader, decltype(&free)>
DataSetUtil::CreateClassificationLabelHeader(
    const std::vector<std::string>& labels) {
  auto header = Malloc<DataSetLabelHeader>(labels.size());
  header->label_array_size = labels.size();
  for (int i = 0; i < labels.size(); i++) {
    const auto& label = labels.at(i);
    CHECK(label.size() < sizeof(header->label_name[0]));
    memset(header->label_name[i], 0, sizeof(header->label_name[0]));
    label.copy(header->label_name[i], label.size(), 0);
  }
  return header;
}

std::unique_ptr<DataItem, decltype(&free)> DataSetUtil::CreateDataItem(
    const DataSetFeatureHeader& header) {
  size_t elem_cnt = header.ElementCount();
  auto body = Malloc<DataItem>(elem_cnt);
  body->len = elem_cnt;
  return std::move(body);
}

std::unique_ptr<DataItem, decltype(&free)> DataSetUtil::CreateImageDataItem(
    const DataSetFeatureHeader& header, const std::string& img_file_path) {
  uint32_t width, height;
  width = header.dim_vec[1];
  height = header.dim_vec[2];
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

std::unique_ptr<DataSetLabel, decltype(&free)> DataSetUtil::CreateDataSetLabel(
    const std::vector<uint32_t>& item_label_indexes) {
  auto body = Malloc<DataSetLabel>(item_label_indexes.size());
  body->len = item_label_indexes.size();
  for (int i = 0; i < body->len; i++) {
    body->data_item_label_idx[i] = item_label_indexes[i];
  }
  return std::move(body);
}

}  // namespace oneflow
