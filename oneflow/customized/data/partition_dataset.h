#ifndef ONEFLOW_CUSTOMIZED_DATA_PARTITION_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_PARTITION_DATASET_H_

#include "oneflow/customized/data/dataset.h"

namespace oneflow {

template<typename LoadTarget>
class PartitionDataset final : public Dataset<LoadTarget> {
 public:
  using BaseDataset = Dataset<LoadTarget>;
  using BaseDatasetUnqPtr = std::unique_ptr<BaseDataset>;
  using LoadTargetShdPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  PartitionDataset(int32_t parallel_num, int32_t parallel_id, BaseDatasetUnqPtr&& dataset)
      : base_(std::move(dataset)),
        num_replicas_(parallel_num),
        rank_(parallel_id),
        cur_idx_(parallel_id) {
    CHECK(base_->EnableRandomAccess());
  }
  ~PartitionDataset() = default;

  LoadTargetShdPtrVec Next() override {
    LoadTargetShdPtrVec ret;
    LoadTargetShdPtr sample = base_->At(cur_idx_);
    ret.push_back(std::move(sample));
    cur_idx_ += num_replicas_;
    return ret;
  }

  bool EnableRandomAccess() override { return false; }
  bool EnableGetSize() override { return false; }

 private:
  BaseDatasetUnqPtr base_;
  int32_t num_replicas_;
  int32_t rank_;
  int64_t cur_idx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_PARTITION_DATASET_H_
