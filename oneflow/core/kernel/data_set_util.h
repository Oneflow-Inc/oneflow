#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/data_set_format.h"

namespace oneflow {

class DataSetUtil final {
 public:
  DataSetUtil() = delete;

  template<typename T>
  static std::unique_ptr<T, decltype(&free)> Malloc(size_t len) {
    T* ptr = reinterpret_cast<T*>(malloc(FlexibleSizeOf<T>(len)));
    return std::unique_ptr<T, decltype(&free)>(ptr, &free);
  }

  static std::unique_ptr<DataSetHeader> CreateHeader(
      const std::string& type, uint32_t data_item_count,
      const std::vector<uint32_t>& dim_array);

  static std::unique_ptr<DataSetLabelDesc, decltype(&free)> CreateLabelDesc(
      const std::vector<std::string>& labels);

  static std::unique_ptr<DataItem, decltype(&free)> CreateDataItem(
      const DataSetHeader& header);

  static std::unique_ptr<DataItem, decltype(&free)> CreateLabelItem(
      const DataSetHeader& header, const std::vector<uint32_t>& label);

  static std::unique_ptr<DataItem, decltype(&free)> CreateImageItem(
      const DataSetHeader& header, const std::string& img_file_path);

 private:
  static void LoadImageData(
      DataItem* body, uint32_t width, uint32_t height,
      const std::function<double(uint32_t d, uint32_t r, uint32_t c)>& Get);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_
