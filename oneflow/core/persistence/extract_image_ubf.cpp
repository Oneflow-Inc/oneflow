#include <gflags/gflags.h>
#include <glog/logging.h>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/normal_ubf_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

DEFINE_string(label_file, "./labels", "label file");
DEFINE_string(feature_file, "./features", "feature file");
DEFINE_string(output_dir, "./", "output direction");
DEFINE_int32(limit, INT32_MAX, "image count limit");

namespace oneflow {
namespace {

void ExtractImage(int num) {
  NormalUbfInStream label_stream(LocalFS(), FLAGS_label_file);
  auto label_header = label_stream.header();
  CHECK(label_header->dim_array_size() == 1);
  std::vector<uint32_t> item_idx2label_idx(label_header->ubf_item_num());
  std::unique_ptr<UbfItem> ubf_item;
  std::set<uint32_t> label_indexes;
  for (int i = 0; label_stream.ReadOneItem(&ubf_item) >= 0; ++i) {
    uint32_t label_idx = reinterpret_cast<const uint32_t*>(ubf_item->body())[0];
    item_idx2label_idx[i] = label_idx;
    label_indexes.insert(label_idx);
  }

  for (uint32_t label_idx : label_indexes) {
    LocalFS()->CreateDirIfNotExist(
        JoinPath(FLAGS_output_dir, std::to_string(label_idx)));
  }

  NormalUbfInStream feature_stream(LocalFS(), FLAGS_feature_file);
  auto feature_header = feature_stream.header();
  CHECK(feature_header->dim_array_size() == 3);
  CHECK(feature_header->ubf_item_num() == label_header->ubf_item_num());
  for (int i = 0; i < num && feature_stream.ReadOneItem(&ubf_item) >= 0; ++i) {
    std::string file_path =
        JoinPath(FLAGS_output_dir, std::to_string(item_idx2label_idx[i]),
                 RemoveExtensionIfExist(Basename(ubf_item->data_id()),
                                        {"JPG", "jpg", "JPEG", "jpeg"})
                     + ".jpg");
    PersistentOutStream out_stream(LocalFS(), file_path);
    out_stream.Write(ubf_item->body(), ubf_item->body_len());
  }
}

}  // namespace
}  // namespace oneflow

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::ExtractImage(FLAGS_limit);
}
