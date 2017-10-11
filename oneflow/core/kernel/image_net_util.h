#ifndef ONEFLOW_CORE_KERNEL_IMAGE_NET_UTIL_H_
#define ONEFLOW_CORE_KERNEL_IMAGE_NET_UTIL_H_

#include "oneflow/core/kernel/data_set_util.h"

namespace oneflow {

class ImageNetUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ImageNetUtil);
  ImageNetUtil() = delete;
  static void CreateDataSetFiles(
      const std::list<std::string>& image_directories,
      const std::string& output_dir) {
    return CreateDataSetFiles(image_directories, output_dir, GlobalFS());
  }
  static void CreateDataSetFiles(
      const std::list<std::string>& image_directories,
      const std::string& output_dir, fs::FileSystem* fs);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_IMAGE_NET_UTIL_H_
