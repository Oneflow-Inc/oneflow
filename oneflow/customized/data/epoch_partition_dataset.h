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

  EpochPartitionDataset(int32_t parallel_num, int32_t parallel_id, bool shuffle,
                        int64_t random_seed, BaseDatasetUnqPtr&& dataset)
      : base_(std::move(dataset)),
        shuffle_(shuffle),
        random_engine_(random_seed),
        num_partitions_(parallel_num),
        pos_(parallel_id) {
    CHECK(base_->EnableRandomAccess());
    CHECK(base_->EnableGetSize());
    index_seq_.resize(base_->Size());
    GenNewIndexSequence();
  }
  ~EpochPartitionDataset() = default;

  LoadTargetShdPtrVec Next() override {
    LoadTargetShdPtrVec ret;
    if (pos_ >= index_seq_.size()) {
      pos_ %= index_seq_.size();
      GenNewIndexSequence();
    }
    LoadTargetShdPtr sample = base_->At(index_seq_.at(pos_));
    ret.push_back(std::move(sample));
    pos_ += num_partitions_;
    return ret;
  }

  bool EnableRandomAccess() override { return false; }
  bool EnableGetSize() override { return false; }

 private:
  void GenNewIndexSequence() {
    std::iota(index_seq_.begin(), index_seq_.end(), 0);
    if (shuffle_) { std::shuffle(index_seq_.begin(), index_seq_.end(), random_engine_); }
  }

  BaseDatasetUnqPtr base_;
  bool shuffle_;
  std::mt19937 random_engine_;
  int32_t num_partitions_;
  int64_t pos_;
  std::vector<int64_t> index_seq_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_EPOCH_PARTITION_DATASET_H_
