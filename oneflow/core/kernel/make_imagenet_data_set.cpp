#include <gflags/gflags.h>
#include <glog/logging.h>
#include "oneflow/core/kernel/imagenet_util.h"
namespace oneflow {}

DEFINE_int32(width, 256, "resized width");
DEFINE_int32(height, 256, "resized height");
DEFINE_string(output_dir, "./", "output direction");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> image_directories;
  for (int i = 1; i < argc; ++i) {
    image_directories.push_back(std::string(argv[i]));
  }
  CHECK(FLAGS_output_dir.size());
  std::unique(image_directories.begin(), image_directories.end());
  oneflow::ImageNetUtil::CreateDataSetFiles(image_directories, FLAGS_width,
                                            FLAGS_height, FLAGS_output_dir);
  return 0;
}
