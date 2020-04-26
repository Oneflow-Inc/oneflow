#ifndef ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

struct COCOImage {
  TensorBuffer data;
  int64_t id;
  int32_t height;
  int32_t width;
};

class COCOMeta;

class COCODataset final : public Dataset<COCOImage> {
 public:
  using LoadTargetShdPtr = std::shared_ptr<COCOImage>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  COCODataset(user_op::KernelInitContext* ctx, COCOMeta* meta);
  ~COCODataset() = default;

  LoadTargetShdPtrVec Next() override;
  LoadTargetShdPtr At(int64_t idx) override;
  int64_t Size() override;

  bool EnableRandomAccess() override { return true; }
  bool EnableGetSize() override { return true; }

 private:
  std::unique_ptr<EmptyTensorManager<COCOImage>> empty_tensor_mgr_;
  const COCOMeta* meta_;
  int64_t cur_idx_;
};

class COCOImageManager final : public EmptyTensorManager<COCOImage> {
 public:
  COCOImageManager(int64_t total_empty_size, int32_t tensor_init_bytes)
      : EmptyTensorManager<COCOImage>(total_empty_size, tensor_init_bytes){};

 protected:
  void PrepareEmpty(COCOImage& image) override {
    image.data.Resize({GetTenosrInitBytes()}, DataType::kChar);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
