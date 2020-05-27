#ifndef ONEFLOW_CUSTOMIZED_DATA_GROUP_BATCH_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_GROUP_BATCH_DATASET_H_

#include "oneflow/customized/data/dataset.h"

namespace oneflow {
namespace data {

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
      : base_(std::move(dataset)),
        batch_size_(batch_size),
        group_fn_(GroupId4Sample),
        order_count_(0) {
    size_t max_group_size = 0;
    while (max_group_size < batch_size) {
      LoadTargetShdPtr data_ptr;
      int64_t group_id = ReadNextDataAndGetGroup(data_ptr);
      auto it = group_id2buffered_samples_.find(group_id);
      if (it == group_id2buffered_samples_.end()) {
        it = group_id2buffered_samples_.emplace(group_id, std::vector<Sample>()).first;
      }
      Sample sample;
      sample.data.swap(data_ptr);
      sample.order = order_count_++;
      it->second.emplace_back(std::move(sample));
      max_group_size = std::max(max_group_size, it->second.size());
    }
  }
  ~GroupBatchDataset() = default;

  LoadTargetShdPtrVec Next() override {
    LoadTargetShdPtrVec ret;
    ret.reserve(batch_size_);
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
    auto group_it = group_id2buffered_samples_.find(group_id);
    if (group_it != group_id2buffered_samples_.end()) {
      auto& buffered_samples = group_it->second;
      auto sample_it = buffered_samples.begin();
      while (remain_cnt > 0 && sample_it != buffered_samples.end()) {
        ret.emplace_back(sample_it->data);
        ++sample_it;
        remain_cnt -= 1;
      }
      buffered_samples.erase(buffered_samples.begin(), sample_it);
    }
    while (remain_cnt > 0) {
      LoadTargetShdPtr data_ptr;
      int64_t cur_group_id = ReadNextDataAndGetGroup(data_ptr);
      if (group_id == -1 || cur_group_id == group_id) {
        ret.emplace_back(std::move(data_ptr));
        if (group_id == -1) { group_id = cur_group_id; }
        remain_cnt -= 1;
      } else {
        auto cur_group_it = group_id2buffered_samples_.find(cur_group_id);
        if (cur_group_it == group_id2buffered_samples_.end()) {
          cur_group_it =
              group_id2buffered_samples_.emplace(cur_group_id, std::vector<Sample>()).first;
        }
        Sample sample;
        sample.data.swap(data_ptr);
        sample.order = order_count_;
        cur_group_it->second.emplace_back(std::move(sample));
      }
      order_count_ += 1;
    }
    return ret;
  }

 private:
  int64_t ReadNextDataAndGetGroup(LoadTargetShdPtr& data_ptr) {
    LoadTargetShdPtrVec data_ptr_vec = base_->Next();
    CHECK_EQ(data_ptr_vec.size(), 1);
    int64_t group_id = group_fn_(data_ptr_vec[0]);
    CHECK_GE(group_id, 0);
    data_ptr.swap(data_ptr_vec[0]);
    return group_id;
  }

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

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_GROUP_BATCH_DATASET_H_
