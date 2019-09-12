#ifndef ONEFLOW_CORE_DATASET_WIDE_DEEP_DATASET_H_
#define ONEFLOW_CORE_DATASET_WIDE_DEEP_DATASET_H_

#include "oneflow/core/data/dataset.h"
#include "json.hpp"

namespace oneflow {
namespace data {

class WideDeepDataset final : public Dataset {
 public:
  explicit WideDeepDataset(const DatasetProto& proto) : Dataset(proto) {}
  virtual ~WideDeepDataset() = default;

  void Init() override;
  size_t Size() const override { return image_ids_.size(); }
  std::unique_ptr<OFRecord> EncodeOneRecord(int64_t idx) const override;
  DataInstance GetDataInstance(int64_t idx) const override;

 private:
  

 private:
  HashMap<std::string, int64_t> feild2field_id_;
  std::vector<int64_t> field_ids_;
  std::vector<int64_t> offset_of_lines_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATASET_WIDE_DEEP_DATASET_H_
