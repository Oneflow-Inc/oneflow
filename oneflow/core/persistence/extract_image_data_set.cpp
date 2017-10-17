#include <gflags/gflags.h>
#include <glog/logging.h>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/imagenet_util.h"
#include "oneflow/core/persistence/normal_data_set_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

DEFINE_string(label_file, "./labels", "label file");
DEFINE_string(feature_file, "./features", "feature file");
DEFINE_string(output_dir, "./", "output direction");

namespace oneflow {
namespace {

void ExtractImage(int num) {
  NormalDataSetInStream label_stream(LocalFS(), FLAGS_label_file);
  NormalDataSetInStream feature_stream(LocalFS(), FLAGS_feature_file);
  auto* label_header = label_stream.header();
  CHECK(label_header->dim_array_size == 1);
  std::vector<uint32_t> item_idx2label_idx(label_header->data_item_count);
  auto buffer = FlexibleMalloc<Buffer>(0);
  for (int i = 0; label_stream.ReadBuffer(&buffer) >= 0; ++i) {
    uint32_t label_idx = reinterpret_cast<uint32_t*>(buffer->data)[0];
    item_idx2label_idx[i] = label_idx;
    LocalFS()->CreateDirIfNotExist(
        JoinPath(FLAGS_output_dir, std::to_string(label_idx)));
  }
  auto* feature_header = feature_stream.header();
  CHECK(feature_header->dim_array_size == 3);
  CHECK(feature_header->data_item_count == label_header->data_item_count);
  for (int i = 0; i < num && feature_stream.ReadBuffer(&buffer) >= 0; ++i) {
    std::string file_path =
        JoinPath(FLAGS_output_dir, std::to_string(item_idx2label_idx[i]),
                 std::to_string(i), ".jpg");
    PersistentOutStream out_stream(LocalFS(), file_path);
    out_stream.Write(buffer->data, buffer->len);
  }
}

}  // namespace
}  // namespace oneflow

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::ExtractImage(3);
}
