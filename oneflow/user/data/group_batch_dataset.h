/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_USER_DATA_GROUP_BATCH_DATASET_H_
#define ONEFLOW_USER_DATA_GROUP_BATCH_DATASET_H_

#include "oneflow/user/data/dataset.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
class GroupBatchDataset final : public Dataset<LoadTarget> {
 public:
  using Base = Dataset<LoadTarget>;
  using SampleType = typename Base::SampleType;
  using BatchType = typename Base::BatchType;
  using NestedDS = Dataset<LoadTarget>;

  GroupBatchDataset(size_t batch_size,
                    const std::function<int64_t(const SampleType&)>& GroupId4Sample,
                    std::unique_ptr<NestedDS>&& dataset)
      : nested_ds_(std::move(dataset)),
        batch_size_(batch_size),
        group_fn_(GroupId4Sample),
        order_count_(0) {}
  ~GroupBatchDataset() = default;

  BatchType Next() override {
    BatchType batch;
    int64_t group_id = FindEarliestBatchGroupId();
    auto group_it = group_id2buffered_samples_.find(group_id);
    if (group_it != group_id2buffered_samples_.end()) {
      auto& batch_sample_list = group_it->second;
      if (!batch_sample_list.empty()) {
        std::swap(batch, batch_sample_list.front().data);
        batch_sample_list.pop_front();
      }
    }
    while (batch.size() < batch_size_) {
      auto next_batch = nested_ds_->Next();
      CHECK_EQ(next_batch.size(), 1);
      int64_t next_group_id = group_fn_(next_batch[0]);
      if (group_id == -1) { group_id = next_group_id; }
      if (group_id == next_group_id) {
        batch.emplace_back(std::move(next_batch[0]));
      } else {
        auto group_it = group_id2buffered_samples_.find(next_group_id);
        if (group_it == group_id2buffered_samples_.end()) {
          group_it =
              group_id2buffered_samples_.emplace(next_group_id, std::list<BatchSample>()).first;
        }
        auto& batch_sample_list = group_it->second;
        if (batch_sample_list.empty() || batch_sample_list.back().data.size() == batch_size_) {
          BatchSample batch_sample;
          std::swap(batch_sample.data, next_batch);
          batch_sample.data.reserve(batch_size_);
          batch_sample.order = order_count_++;
          batch_sample_list.emplace_back(std::move(batch_sample));
        } else {
          batch_sample_list.back().data.emplace_back(std::move(next_batch[0]));
        }
      }
    }
    return batch;
  }

 private:
  int64_t FindEarliestBatchGroupId() const {
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
    return group_id;
  }

  struct BatchSample {
    BatchType data;
    int64_t order;
  };

  std::unique_ptr<NestedDS> nested_ds_;
  size_t batch_size_;
  std::function<int64_t(const SampleType&)> group_fn_;
  std::map<int64_t, std::list<BatchSample>> group_id2buffered_samples_;
  int64_t order_count_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_GROUP_BATCH_DATASET_H_
