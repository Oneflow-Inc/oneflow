#include "oneflow/customized/data/coco_dataset.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

COCODataset::COCODataset(user_op::KernelInitContext* ctx) {
  // Read content of annotation file (json format) to json obj
  const std::string& anno_path = ctx->GetAttr<std::string>("annotation_file");
  PersistentInStream in_stream(DataFS(), anno_path);
  std::string json_str;
  std::string line;
  while (in_stream.ReadLine(&line) == 0) { json_str += line; }
  std::istringstream in_str_stream(json_str);
  in_str_stream >> annotation_json_;
}

COCODataset::LoadTargetPtrList COCODataset::Next() { TODO(); }

COCODataset::LoadTargetPtr COCODataset::At(int64_t idx) { TODO(); }

}  // namespace oneflow
