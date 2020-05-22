#ifndef ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/customized/data/empty_tensor_manager.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace data {

struct COCOImage {
  TensorBuffer data;
  int64_t index;
  int64_t id;
  int32_t height;
  int32_t width;
};

class COCOMeta;

class COCODataset final : public RandomAccessDataset<COCOImage> {
 public:
  using LoadTargetShdPtr = std::shared_ptr<COCOImage>;

  COCODataset(user_op::KernelInitContext* ctx, const std::shared_ptr<const COCOMeta>& meta);
  ~COCODataset() = default;

  LoadTargetShdPtr At(int64_t index) const override;
  size_t Size() const override;

 private:
  std::unique_ptr<EmptyTensorManager<COCOImage>> empty_tensor_mgr_;
  std::shared_ptr<const COCOMeta> meta_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
