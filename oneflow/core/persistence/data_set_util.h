#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/data_set_format.h"

namespace oneflow {

class DataSetUtil final {
 public:
  DataSetUtil() = delete;

  static uint32_t ValidateHeader(const DataSetHeader& header);
  static void UpdateHeaderCheckSum(DataSetHeader* header);

  static std::unique_ptr<Record, decltype(&free)> NewRecord(
      const std::string& key, size_t value_buf_len, DataType dtype,
      DataCompressType dctype, const std::function<void(char* buff)>& Fill);

  static uint8_t ValidateRecord(const Record& buff);

  static std::unique_ptr<Record, decltype(&free)> NewRecord(
      const std::string& key, size_t value_buf_len, DataType dtype,
      const std::function<void(char* buff)>& Fill) {
    return NewRecord(key, value_buf_len, dtype, DataCompressType::kNoCompress,
                     Fill);
  }

  static std::unique_ptr<DataSetHeader> CreateHeader(
      const std::string& type, uint32_t data_item_count,
      const std::vector<uint32_t>& dim_array);

  static std::unique_ptr<Record, decltype(&free)> CreateDataItem(
      const DataSetHeader& header);

  static std::unique_ptr<Record, decltype(&free)> CreateLabelItem(
      const DataSetHeader& header, const std::string& key, uint32_t label);

  static std::unique_ptr<Record, decltype(&free)> CreateImageItem(
      const DataSetHeader& header, const std::string& img_file_path);

  static void ExtractImage(const Record& data_item, const DataSetHeader& header,
                           const std::string& output_img_path);

 private:
  static uint8_t ValidateRecordMeta(const Record& buffer);
  static void UpdateRecordCheckSum(Record* buffer);
  static void UpdateRecordMetaCheckSum(Record* buffer);
  static void LoadImageData(
      Record* body, uint32_t width, uint32_t height,
      const std::function<double(uint32_t d, uint32_t r, uint32_t c)>& Get);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_UTIL_H_
