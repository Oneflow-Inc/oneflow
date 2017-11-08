#include <gflags/gflags.h>
#include <glog/logging.h>
#include "oneflow/core/persistence/ubf_util.h"
namespace oneflow {}

DEFINE_int32(limit, INT_MAX, "packed image count limit");
DEFINE_int32(width, 256, "resized width");
DEFINE_int32(height, 256, "resized height");
DEFINE_string(output_dir, "./", "output direction");
DEFINE_bool(use_hadoop_stream, false, "use hadoop stream file as input");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> image_directories;
  for (int i = 1; i < argc; ++i) {
    image_directories.push_back(std::string(argv[i]));
    std::cout << argv[i] << std::endl;
  }
  CHECK(FLAGS_output_dir.size());
  std::unique(image_directories.begin(), image_directories.end());
  oneflow::UbfUtil::CreateUbfFiles(image_directories, FLAGS_limit, FLAGS_width,
                                   FLAGS_height, FLAGS_output_dir,
                                   FLAGS_use_hadoop_stream);
  return 0;
}
