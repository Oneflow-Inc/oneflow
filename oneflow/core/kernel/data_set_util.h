#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_

#include <opencv2/opencv.hpp>
#include "oneflow/core/common/util.h"
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

  static size_t DataSetFeatureHeaderSize(uint32_t dim_num);

  static std::unique_ptr<DataSetHeaderDesc> CreateHeaderDesc(
      const DataSetLabelHeader& header, uint32_t data_item_size);

  static std::unique_ptr<DataSetHeaderDesc> CreateHeaderDesc(
      const DataSetFeatureHeader& header, uint32_t data_item_size);

  static std::unique_ptr<DataSetFeatureHeader, decltype(free) *>
  CreateImageFeatureHeader(uint32_t width, uint32_t height);

  static std::unique_ptr<DataSetLabelHeader, decltype(free) *>
  CreateClassificationLabelHeader(const std::vector<std::string>& labels);

  static std::unique_ptr<DataItem, decltype(free) *> CreateDataItem(
      const DataSetFeatureHeader& header);

  static std::unique_ptr<DataItem, decltype(free) *> CreateImageDataItem(
      const DataSetFeatureHeader& header, const std::string& img_file_path);

  static void CreateImageColorValueGetter(
      const cv::Mat& img,
      std::function<double(uint32_t d, uint32_t row, uint32_t col)>* getter);

  static void LoadImageData(
      DataItem* body, uint32_t width, uint32_t height,
      const std::function<double(uint32_t d, uint32_t r, uint32_t c)>& Get);

  static std::unique_ptr<DataSetLabel, decltype(free) *> CreateDataSetLabel(
      const std::vector<uint32_t>& item_label_indexes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_
