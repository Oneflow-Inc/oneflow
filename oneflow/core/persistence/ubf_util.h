#ifndef ONEFLOW_CORE_PERSISTENCE_DATA_SET_UTIL_H_
#define ONEFLOW_CORE_PERSISTENCE_DATA_SET_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/ubf_header.h"
#include "oneflow/core/persistence/ubf_item.h"

namespace oneflow {

class UbfUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UbfUtil);
  UbfUtil() = delete;

  static std::unique_ptr<UbfItem, decltype(&free)> CreateLabelItem(
      const std::string& key, uint32_t label);

  static std::unique_ptr<UbfItem, decltype(&free)> CreateImageItem(
      const std::string& img_file_path);

  static void ExtractImage(const UbfItem& data_item, const UbfHeader& header,
                           const std::string& output_img_path);
  static void CreateUbfFiles(const std::vector<std::string>& image_directories,
                             uint32_t limit, uint32_t width, uint32_t height,
                             const std::string& output_dir);

 private:
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
