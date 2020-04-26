#ifndef ONEFLOW_CUSTOMIZED_DATA_GROUP_BATCH_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_GROUP_BATCH_DATASET_H_

#include "oneflow/customized/data/dataset.h"

namespace oneflow {

template<typename LoadTarget>
class GroupBatchDataset final : public Dataset<LoadTarget> {
 public:
  using BaseDataset = Dataset<LoadTarget>;
  using BaseDatasetUnqPtr = std::unique_ptr<BaseDataset>;
  using LoadTargetShdPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  GroupBatchDataset(size_t batch_size,
                    const std::function<int64_t(const LoadTargetShdPtr&)>& GroupId4Sample,
                    BaseDatasetUnqPtr&& dataset)
      : base_(std::move(dataset)), group_fn_(GroupId4Sample) {}
  ~GroupBatchDataset() = default;

  LoadTargetShdPtrVec Next() override {
    LoadTargetShdPtrVec ret;
    size_t remain_cnt = batch_size_;
    // decide batch group_id according to first sample whether from cache or base
    int64_t group_id = -1;
    if (cache_samples_.size() > 0) {
      group_id = cache_group_ids_.front();
      ret.emplace_back(cache_samples_.front());
      cache_group_ids_.pop_front();
      cache_samples_.pop_front();
    } else {
      LoadTargetShdPtrVec samples = base_->Next();
      CHECK_EQ(samples.size(), 1);
      group_id = group_fn_(samples[0]);
      ret.emplace_back(std::move(samples[0]));
    }
    remain_cnt -= 1;
    // fetch all samples with the same group_id with batch from cache
    auto group_it = cache_group_ids_.begin();
    auto sample_it = cache_samples_.begin();
    while (group_it != cache_group_ids_.end()) {
      if (*group_it == group_id) {
        ret.emplace_back(*sample_it);
        remain_cnt -= 1;
        group_it = cache_group_ids_.erase(group_it);
        sample_it = cache_samples_.erase(sample_it);
      } else {
        ++group_it;
        ++sample_it;
      }
    }
    // fetch samples from base, samples with the same group join batch,
    // otherwise cache it
    while (remain_cnt > 0) {
      LoadTargetShdPtrVec samples = base_->Next();
      CHECK_EQ(samples.size(), 1);
      int64_t cur_group_id = group_fn_(samples[0]);
      if (cur_group_id == group_id) {
        ret.emplace_back(std::move(samples[0]));
        remain_cnt -= 1;
      } else {
        cache_samples_.emplace_back(std::move(samples[0]));
        cache_group_ids_.emplace_back(cur_group_id);
      }
    }
    return ret;
  }

  bool EnableRandomAccess() override { return false; }
  bool EnableGetSize() override { return false; }

 private:
  BaseDatasetUnqPtr base_;
  size_t batch_size_;
  std::function<int64_t(const LoadTargetShdPtr&)> group_fn_;
  std::list<LoadTargetShdPtr> cache_samples_;
  std::list<int64_t> cache_group_ids_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_GROUP_BATCH_DATASET_H_
