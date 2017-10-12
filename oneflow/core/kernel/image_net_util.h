#ifndef ONEFLOW_CORE_KERNEL_IMAGE_NET_UTIL_H_
#define ONEFLOW_CORE_KERNEL_IMAGE_NET_UTIL_H_

#include <list>
#include <string>
#include "oneflow/core/kernel/data_set_util.h"

namespace oneflow {

class ImageNetUtil final {
 public:
  ImageNetUtil() = delete;
  static void CreateDataSetFiles(
      const std::list<std::string>& image_directories, uint32_t width,
      uint32_t height, const std::string& output_dir);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_IMAGE_NET_UTIL_H_
