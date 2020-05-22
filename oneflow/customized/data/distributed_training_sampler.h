#ifndef ONEFLOW_CUSTOMIZED_DATA_DISTRIBUTED_TRAINING_SAMPLER_H_
#define ONEFLOW_CUSTOMIZED_DATA_DISTRIBUTED_TRAINING_SAMPLER_H_

#include "oneflow/customized/data/dataset.h"

namespace oneflow {
namespace data {

class DistributedTrainingSampler final : public Sampler {
 public:
  DistributedTrainingSampler(int64_t epoch_size, int64_t parallel_num, int64_t parallel_id,
                             bool stride_partition, bool shuffle, int64_t random_seed)
      : shuffle_(shuffle),
        stride_partition_(stride_partition),
        rnd_seed_(random_seed),
        num_shards_(parallel_num),
        pos_(0),
        pos_in_shard_(0),
        epoch_cnt_(0) {
    if (rnd_seed_ == -1) { rnd_seed_ = NewRandomSeed(); }
    if (stride_partition) { pos_ = parallel_id; }
    shard_size_ = std::ceil(static_cast<float>(epoch_size) / num_shards_);
    index_seq_.resize(epoch_size);
    std::iota(index_seq_.begin(), index_seq_.end(), 0);
    GenNewIndexSequence();
  }
  virtual ~DistributedTrainingSampler() = default;

  int64_t Next() override {
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
    return index_seq_.at(pos_);
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

#endif  // ONEFLOW_CUSTOMIZED_DATA_DISTRIBUTED_TRAINING_SAMPLER_H_
