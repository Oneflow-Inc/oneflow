#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_

#include <glog/logging.h>
#include <memory>
#include "oneflow/core/kernel/data_set_format.h"

namespace oneflow {

class DataSetUtil final {
 public:
  DataSetUtil() = delete;

  template<typename T>
  static std::unique_ptr<T, decltype(free) *> Malloc(size_t len) {
    T* ptr = reinterpret_cast<T*>(malloc(len));
    return std::unique_ptr<T, decltype(free)*>(ptr, &free);
  }

  static size_t DataSetFeatureHeaderSize(uint32_t dim_num) {
    DataSetFeatureHeader* ptr = nullptr;
    return sizeof(ptr->dim_array_size) + dim_num * sizeof(ptr->dim_vec[0]);
  }

  static std::unique_ptr<DataSetHeaderDesc> CreateHeaderDesc(
      const DataSetLabelHeader& header, uint32_t data_item_size) {
    std::unique_ptr<DataSetHeaderDesc> desc(new DataSetHeaderDesc);
    strcpy(desc->type, "label");
    desc->data_item_size = data_item_size;
    desc->header_buffer_len = header.Size();
    return std::move(desc);
  }

  static std::unique_ptr<DataSetHeaderDesc> CreateHeaderDesc(
      const DataSetFeatureHeader& header, uint32_t data_item_size) {
    std::unique_ptr<DataSetHeaderDesc> desc(new DataSetHeaderDesc);
    strcpy(desc->type, "feature");
    desc->data_item_size = data_item_size;
    desc->header_buffer_len = header.Size();
    return std::move(desc);
  }

  static std::unique_ptr<DataSetFeatureHeader, decltype(free) *>
  CreateImageFeatureHeader(uint32_t width, uint32_t height) {
    auto header = Malloc<DataSetFeatureHeader>(3);
    header->dim_array_size = 3;
    header->dim_vec[0] = 3;
    header->dim_vec[1] = width;
    header->dim_vec[2] = height;
    return std::move(header);
  }

  static std::unique_ptr<DataSetLabelHeader, decltype(free) *>
  CreateClassificationLabelHeader(const std::vector<std::string>& labels) {
    auto header = Malloc<DataSetLabelHeader>(labels.size());
    header->label_array_size = labels.size();
    for (int i = 0; i < labels.size(); i++) {
      const auto& label = labels[i];
      CHECK(label.size() < sizeof(header->label_name[0]));
      strncpy(header->label_name[i], label.c_str(),
              sizeof(header->label_name[0]) - 1);
    }
    return header;
  }

  static std::unique_ptr<DataItem, decltype(free) *> CreateDataItem(
      const DataSetFeatureHeader& header) {
    size_t elem_cnt = header.ElementCount();
    auto body = Malloc<DataItem>(elem_cnt);
    body->len = elem_cnt;
    return std::move(body);
  }

  static void LoadImageData(
      DataItem* body, uint32_t width, uint32_t height,
      const std::function<double(uint32_t w, uint32_t h)>& GetRed,
      const std::function<double(uint32_t w, uint32_t h)>& GetGreen,
      const std::function<double(uint32_t w, uint32_t h)>& GetBlue) {
    uint32_t image_size = width * height;
    CHECK(image_size * 3 == body->len);
#define LOAD_IMAGE_CHANNEL(get_color_value, depth) \
  for (uint32_t i = 0; i < image_size; i++) {      \
    body->data[i + depth * image_size] =           \
        get_color_value(i / height, i % height);   \
  }

    LOAD_IMAGE_CHANNEL(GetRed, 0);
    LOAD_IMAGE_CHANNEL(GetGreen, 1);
    LOAD_IMAGE_CHANNEL(GetBlue, 2);
  }

  static std::unique_ptr<DataSetLabel, decltype(free) *> CreateDataSetLabel(
      const std::vector<uint32_t>& item_label_indexes) {
    auto body = Malloc<DataSetLabel>(item_label_indexes.size());
    body->len = item_label_indexes.size();
    for (int i = 0; i < body->len; i++) {
      body->data_item_label_idx[i] = item_label_indexes[i];
    }
    return std::move(body);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_
