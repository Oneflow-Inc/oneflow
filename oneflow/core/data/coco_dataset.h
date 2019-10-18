#ifndef ONEFLOW_CORE_DATA_COCO_DATASET_H_
#define ONEFLOW_CORE_DATA_COCO_DATASET_H_

#include "oneflow/core/data/dataset.h"
#include <json.hpp>

namespace oneflow {
namespace data {

class COCODataset final : public Dataset {
 public:
  explicit COCODataset(const DatasetProto& proto);
  virtual ~COCODataset() override = default;

  size_t Size() const override { return image_ids_.size(); }
  void GetData(int64_t idx, DataInstance* data_inst) const;
  int64_t GetGroupId(int64_t idx) const override;

 private:
  bool ImageHasValidAnnotations(int64_t image_id) const;
  void GetImage(const nlohmann::json& image_json, DataField* image_field,
                DataField* image_size_field) const;
  void GetBbox(const nlohmann::json& bbox_json, const nlohmann::json& image_json,
               DataField* data_field) const;
  void GetLabel(const nlohmann::json& label_json, DataField* data_field) const;
  void GetSegmentation(const nlohmann::json& segmentation, DataField* data_field) const;

 private:
  static const int32_t kMinKeypointsPerImage = 10;

  nlohmann::json annotation_json_;
  std::vector<int64_t> image_ids_;
  HashMap<int64_t, const nlohmann::json&> image_id2image_;
  HashMap<int64_t, const nlohmann::json&> anno_id2anno_;
  HashMap<int64_t, std::vector<int64_t>> image_id2anno_id_;
  HashMap<int32_t, int32_t> category_id2contiguous_id_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATA_COCO_DATASET_H_
