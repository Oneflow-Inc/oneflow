#ifndef ONEFLOW_CORE_KERNEL_IMAGE_NET_UTIL_H_
#define ONEFLOW_CORE_KERNEL_IMAGE_NET_UTIL_H_

#include "oneflow/core/kernel/data_set_util.h"

namespace oneflow {

class ImageNetUtil final {
 public:
  ImageNetUtil() = delete;
  static void CreateDataSetFiles(
      const std::vector<std::string>& image_directories, uint32_t width,
      uint32_t height, const std::string& output_dir);

 private:
  static void GetFilePaths(
      const std::vector<std::string>& image_directories,
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

#endif  // ONEFLOW_CORE_KERNEL_IMAGE_NET_UTIL_H_
