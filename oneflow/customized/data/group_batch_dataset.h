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
      : base_(std::move(dataset)), group_fn_(GroupId4Sample), order_count_(0) {}
  ~GroupBatchDataset() = default;

  LoadTargetShdPtrVec Next() override {
    LoadTargetShdPtrVec ret;
    int64_t group_id = -1;
    int64_t min_order = -1;
    for (const auto& pair : group_id2buffered_samples_) {
      if (pair.second.size() > 0) {
        if (min_order == -1 || pair.second.front().order < min_order) {
          min_order = pair.second.front().order;
          group_id = pair.first;
        }
      }
    }
    size_t remain_cnt = batch_size_;
    auto& buffered_samples = group_id2buffered_samples_.at(group_id);
    auto it = buffered_samples.begin();
    while (remain_cnt > 0 && it != buffered_samples.end()) {
      ret.emplace_back(it->data);
      ++it;
      remain_cnt -= 1;
    }
    buffered_samples.erase(buffered_samples.begin(), it);
    while (remain_cnt > 0) {
      LoadTargetShdPtrVec data_inst_vec = base_->Next();
      CHECK_EQ(data_inst_vec.size(), 1);
      int64_t cur_group_id = group_fn_(data_inst_vec[0]);
      if (cur_group_id == group_id) {
        ret.emplace_back(std::move(data_inst_vec[0]));
        remain_cnt -= 1;
      } else {
        auto it = group_id2buffered_samples_.find(cur_group_id);
        if (it == group_id2buffered_samples_.end()) {
          it = group_id2buffered_samples_.emplace(cur_group_id, std::vector<Sample>()).first;
        }
        Sample sample;
        sample.data.swap(data_inst_vec[0]);
        sample.order = order_count_;
        it->second.emplace_back(std::move(sample));
      }
      order_count_ += 1;
    }
    return ret;
  }

  bool EnableRandomAccess() override { return false; }
  bool EnableGetSize() override { return false; }

 private:
  struct Sample {
    LoadTargetShdPtr data;
    int64_t order;
  };

  BaseDatasetUnqPtr base_;
  size_t batch_size_;
  std::function<int64_t(const LoadTargetShdPtr&)> group_fn_;
  std::map<int64_t, std::vector<Sample>> group_id2buffered_samples_;
  int64_t order_count_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_GROUP_BATCH_DATASET_H_
