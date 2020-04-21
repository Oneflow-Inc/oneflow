#ifndef ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"
#include <json.hpp>

namespace oneflow {

struct COCODataInstance {
  TensorBuffer image;
  int64_t image_id;
  int64_t image_height;
  int64_t image_width;
  TensorBuffer object_bboxes;
  TensorBuffer object_labels;
  // ... store other data like bbox , segmentation
};

class COCODataset final : public Dataset<COCODataInstance> {
 public:
  using LoadTargetPtr = std::shared_ptr<COCODataInstance>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  COCODataset(user_op::KernelInitContext* ctx);
  ~COCODataset() = default;

  LoadTargetPtrList Next() override;
  LoadTargetPtr At(int64_t idx) override;

  int64_t Size() override { return image_ids_.size(); }

  bool EnableRandomAccess() override { return true; }
  bool EnableGetSize() override { return true; }

 private:
  nlohmann::json annotation_json_;
  std::vector<int64_t> image_ids_;
  HashMap<int64_t, const nlohmann::json&> image_id2image_;
  HashMap<int64_t, const nlohmann::json&> anno_id2anno_;
  HashMap<int64_t, std::vector<int64_t>> image_id2anno_id_;
  HashMap<int32_t, int32_t> category_id2contiguous_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
