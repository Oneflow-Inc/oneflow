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
#ifndef ONEFLOW_USER_DATA_DISTRIBUTED_TRAINING_DATASET_H_
#define ONEFLOW_USER_DATA_DISTRIBUTED_TRAINING_DATASET_H_

#include "oneflow/user/data/dataset.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
class DistributedTrainingDataset final : public Dataset<LoadTarget> {
 public:
  using Base = Dataset<LoadTarget>;
  using SampleType = typename Base::SampleType;
  using BatchType = typename Base::BatchType;
  using NestedDS = RandomAccessDataset<LoadTarget>;

  DistributedTrainingDataset(int64_t parallel_num, int64_t parallel_id, bool stride_partition,
                             bool shuffle, int64_t random_seed, std::unique_ptr<NestedDS>&& dataset)
      : nested_ds_(std::move(dataset)),
        shuffle_(shuffle),
        stride_partition_(stride_partition),
        rnd_seed_(random_seed),
        num_shards_(parallel_num),
        pos_(0),
        pos_in_shard_(0),
        epoch_cnt_(0) {
    shard_size_ = std::ceil(static_cast<float>(nested_ds_->Size()) / num_shards_);
    if (stride_partition) {
      pos_ = parallel_id;
    } else {
      pos_ = parallel_id * shard_size_;
    }
    index_seq_.resize(nested_ds_->Size());
    std::iota(index_seq_.begin(), index_seq_.end(), 0);
    GenNewIndexSequence();
  }
  virtual ~DistributedTrainingDataset() = default;

  virtual BatchType Next() override {
    // There are 2 partition strategies
    // assume epoch size is 10, index seq don't shuffle and there are 4 parts
    // stride partition strategy (when stride_partition is true):
    //       |  part1   |  part2   |  part3   |  part4   |
    // iter0 | 0, 4, 8, | 1, 5, 9, | 2, 6, 0, | 3, 7, 1, |
    // iter1 | 2, 6, 0, | 3, 7, 1, | 4, 8, 2, | 5, 9, 3, |
    // contiguous partition strategy (when stride_partition is false):
    //       |  part1   |  part2   |  part3   |  part4   |
    // iter0 | 0, 1, 2, | 3, 4, 5, | 6, 7, 8, | 9, 0, 1, |
    // iter1 | 2, 3, 4, | 5, 6, 7, | 8, 9, 0, | 1, 2, 3, |
    BatchType batch = nested_ds_->At(index_seq_.at(pos_));
    if (stride_partition_) {
      pos_ += num_shards_;
    } else {
      pos_ += 1;
      pos_in_shard_ += 1;
      if (pos_in_shard_ == shard_size_) {
        pos_ += (num_shards_ - 1) * shard_size_;
        pos_in_shard_ = 0;
      }
    }
    CheckRanOutOfSize();
    return batch;
  }

 private:
  void CheckRanOutOfSize() {
    if (pos_ >= index_seq_.size()) {
      GenNewIndexSequence();
      pos_ %= index_seq_.size();
    }
  }

  void GenNewIndexSequence() {
    if (shuffle_) {
      std::mt19937 engine(rnd_seed_ + epoch_cnt_);
      std::shuffle(index_seq_.begin(), index_seq_.end(), engine);
    }
    epoch_cnt_ += 1;
  }

  std::unique_ptr<NestedDS> nested_ds_;
  bool shuffle_;
  bool stride_partition_;
  int64_t rnd_seed_;
  int64_t num_shards_;
  int64_t shard_size_;
  int64_t pos_;
  int64_t pos_in_shard_;
  int64_t epoch_cnt_;
  std::vector<int64_t> index_seq_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_DISTRIBUTED_TRAINING_DATASET_H_
