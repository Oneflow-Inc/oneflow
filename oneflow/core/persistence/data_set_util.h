#ifndef ONEFLOW_CORE_PERSISTENCE_DATA_SET_UTIL_H_
#define ONEFLOW_CORE_PERSISTENCE_DATA_SET_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/record.h"

namespace oneflow {

class DataSetUtil final {
 public:
  DataSetUtil() = delete;

  static uint32_t ValidateHeader(const DataSetHeader& header);
  static void UpdateHeaderCheckSum(DataSetHeader* header);

  static std::unique_ptr<Record, decltype(&free)> NewRecord(
      const std::string& key, size_t value_buf_len, DataType dtype,
      DataEncodeType detype, const std::function<void(char* buff)>& Fill);

  static uint8_t ValidateRecord(const Record& buff);

  static std::unique_ptr<Record, decltype(&free)> NewRecord(
      const std::string& key, size_t value_buf_len, DataType dtype,
      const std::function<void(char* buff)>& Fill) {
    return NewRecord(key, value_buf_len, dtype, DataEncodeType::kNoEncode,
                     Fill);
  }

  static std::unique_ptr<DataSetHeader> CreateHeader(
      const std::string& type, uint32_t data_item_count,
      const std::vector<uint32_t>& dim_array);

  static std::unique_ptr<Record, decltype(&free)> CreateLabelItem(
      const std::string& key, uint32_t label);

  static std::unique_ptr<Record, decltype(&free)> CreateImageItem(
      const std::string& img_file_path);

  static void ExtractImage(const Record& data_item, const DataSetHeader& header,
                           const std::string& output_img_path);
  static void CreateDataSetFiles(
      const std::vector<std::string>& image_directories, uint32_t limit,
      uint32_t width, uint32_t height, const std::string& output_dir);

 private:
  static uint8_t ValidateRecordMeta(const Record& buffer);
  static void UpdateRecordCheckSum(Record* buffer);
  static void UpdateRecordMetaCheckSum(Record* buffer);
  static void GetFilePaths(
      const std::vector<std::string>& image_directories, uint32_t limit,
      std::vector<std::string>* img_file_paths,
      std::unordered_map<std::string, uint32_t>* file_path2label_idx);
  static void SaveLabels(
      const std::vector<std::string>& image_directories,
      const std::vector<std::string>& img_file_paths,
      const std::unordered_map<std::string, uint32_t>& file_path2label_idx,
      const std::string& output_dir);
  static void SaveFeatures(const std::vector<std::string>& img_file_paths,
                           uint32_t width, uint32_t height,
                           const std::string& output_dir);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_DATA_SET_UTIL_H_
