#include <gflags/gflags.h>
#include <glog/logging.h>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/kernel/imagenet_util.h"
#include "oneflow/core/kernel/normal_data_set_in_stream.h"

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
  auto* label_desc = label_stream.label_desc();
  std::vector<std::string> label_idx2dir_path(label_desc->label_array_size);
  for (int i = 0; i < label_desc->label_array_size; ++i) {
    CHECK(strlen(label_desc->label_desc[i])
          < sizeof(label_desc->label_desc[0]));
    std::string label_name(basename(label_desc->label_desc[i]));
    label_idx2dir_path[i] = JoinPath(FLAGS_output_dir, label_name);
    LocalFS()->CreateDirIfNotExist(label_idx2dir_path[i]);
  }
  std::vector<uint32_t> item_idx2label_idx(label_header->data_item_count);
  auto data_item = DataSetUtil::Malloc<DataItem>(1);
  for (int i = 0; label_stream.ReadDataItem(&data_item) >= 0; ++i) {
    item_idx2label_idx[i] = data_item->data[0];
  }
  auto* feature_header = feature_stream.header();
  CHECK(feature_header->dim_array_size == 3);
  CHECK(feature_header->data_item_count == label_header->data_item_count);
  for (int i = 0; i < num && feature_stream.ReadDataItem(&data_item) >= 0;
       ++i) {
    std::string file_path =
        JoinPath(label_idx2dir_path[item_idx2label_idx[i]], std::to_string(i));
    DataSetUtil::ExtractImage(*data_item, *feature_header, file_path);
  }
}

}  // namespace
}  // namespace oneflow

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::ExtractImage(3);
}
