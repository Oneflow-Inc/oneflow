#ifndef ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

struct COCODataInstance {
  TensorBuffer image;
  int64_t label;
  int64_t image_height;
  int64_t image_width;
  // ... store other data like bbox , segmentation
};

class COCODataset final : public Dataset<COCODataInstance> {
 public:
  using LoadTargetPtr = std::shared_ptr<COCODataInstance>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  COCODataset(user_op::KernelInitContext* ctx) { TODO(); }
  ~COCODataset() = default;

  LoadTargetPtrList Next() override { TODO(); }

 private:
  // maybe not this member
  std::vector<int64_t> image_ids_;
  // other member list like image name, anno ...
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
