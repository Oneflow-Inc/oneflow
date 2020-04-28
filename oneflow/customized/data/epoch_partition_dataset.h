#ifndef ONEFLOW_CUSTOMIZED_DATA_EPOCH_PARTITION_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_EPOCH_PARTITION_DATASET_H_

#include "oneflow/customized/data/dataset.h"

namespace oneflow {

template<typename LoadTarget>
class EpochPartitionDataset final : public Dataset<LoadTarget> {
 public:
  using BaseDataset = Dataset<LoadTarget>;
  using BaseDatasetUnqPtr = std::unique_ptr<BaseDataset>;
  using LoadTargetShdPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  EpochPartitionDataset(int32_t parallel_num, int32_t parallel_id, bool stride_part, bool shuffle,
                        int64_t random_seed, BaseDatasetUnqPtr&& dataset)
      : base_(std::move(dataset)),
        shuffle_(shuffle),
        seed_(random_seed),
        stride_partition_(stride_part),
        num_partitions_(parallel_num),
        pos_(0),
        pos_in_partition_(0),
        epoch_cnt_(0) {
    CHECK(base_->EnableRandomAccess());
    CHECK(base_->EnableGetSize());
    if (seed_ == -1) { seed_ = NewRandomSeed(); }
    if (stride_part) { pos_ = parallel_id; }
    partition_size_ = std::ceil(static_cast<float>(base_->Size()) / num_partitions_);
    index_seq_.resize(base_->Size());
    std::iota(index_seq_.begin(), index_seq_.end(), 0);
    GenNewIndexSequence();
  }
  ~EpochPartitionDataset() = default;

  LoadTargetShdPtrVec Next() override {
    // 2 partition strategy
    // assume dataset size is 10, index seq don't shuffle and there are 4 parts
    // stride partition strategy:
    //       |  part1   |  part2   |  part3   |  part4   |
    // iter0 | 0, 4, 8, | 1, 5, 9, | 2, 6, 0, | 3, 7, 1, |
    // iter1 | 2, 6, 0, | 3, 7, 1, | 4, 8, 2, | 5, 9, 3, |
    // contiguous partition strategy:
    //       |  part1   |  part2   |  part3   |  part4   |
    // iter0 | 0, 1, 2, | 3, 4, 5, | 6, 7, 8, | 9, 0, 1, |
    // iter1 | 2, 3, 4, | 5, 6, 7, | 8, 9, 0, | 1, 2, 3, |
    LoadTargetShdPtrVec ret;
    CheckRanOutOfSize();
    LoadTargetShdPtr sample = base_->At(index_seq_.at(pos_));
    ret.push_back(std::move(sample));
    if (stride_partition_) {
      pos_ += num_partitions_;
    } else {
      pos_ += 1;
      pos_in_partition_ += 1;
      if (pos_in_partition_ == partition_size_) {
        pos_ += (num_partitions_ - 1) * partition_size_;
        CheckRanOutOfSize();
        pos_in_partition_ = 0;
      }
    }
    return ret;
  }

  bool EnableRandomAccess() override { return false; }
  bool EnableGetSize() override { return false; }

 private:
  void CheckRanOutOfSize() {
    if (pos_ >= index_seq_.size()) {
      GenNewIndexSequence();
      pos_ %= index_seq_.size();
    }
  }

  void GenNewIndexSequence() {
    if (shuffle_) {
      std::mt19937 engine(seed_ + epoch_cnt_);
      std::shuffle(index_seq_.begin(), index_seq_.end(), engine);
    }
    epoch_cnt_ += 1;
  }

  BaseDatasetUnqPtr base_;
  bool shuffle_;
  int64_t seed_;
  bool stride_partition_;
  int64_t num_partitions_;
  int64_t partition_size_;
  int64_t pos_;
  int64_t pos_in_partition_;
  int64_t epoch_cnt_;
  std::vector<int64_t> index_seq_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_EPOCH_PARTITION_DATASET_H_
