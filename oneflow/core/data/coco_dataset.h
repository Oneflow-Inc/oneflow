#ifndef ONEFLOW_CORE_DATASET_COCO_DATASET_H_
#define ONEFLOW_CORE_DATASET_COCO_DATASET_H_

#include "oneflow/core/data/dataset.h"
#include "json.hpp"

namespace oneflow {
namespace data {

class COCODataset final : public Dataset {
 public:
  explicit COCODataset(const DatasetProto& proto) : Dataset(proto) {}
  virtual ~COCODataset() = default;

  void Init() override;
  size_t Size() const override { return image_ids_.size(); }
  std::unique_ptr<OFRecord> EncodeOneRecord(int64_t idx) const override;

 private:
  void EncodeImage(int64_t image_id, Feature& feature) const;
  void EncodeSegmentation(int64_t image_id, Feature& feature) const;

 private:
  nlohmann::json annotation_json_;
  std::vector<int64_t> image_ids_;
  HashMap<int64_t, const nlohmann::json&> image_id2image_;
  HashMap<int64_t, const nlohmann::json&> anno_id2anno_;
  HashMap<int64_t, std::vector<int64_t>> image_id2anno_id_;
  HashMap<int32_t, int32_t> category_id2contiguous_id_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATASET_COCO_DATASET_H_
