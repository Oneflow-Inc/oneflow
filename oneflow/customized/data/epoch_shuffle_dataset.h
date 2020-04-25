#ifndef ONEFLOW_CUSTOMIZED_DATA_EPOCH_SHUFFLE_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_EPOCH_SHUFFLE_DATASET_H_

#include "oneflow/customized/data/dataset.h"

namespace oneflow {

template<typename LoadTarget>
class EpochShuffleDataset final : public Dataset<LoadTarget> {
 public:
  using BaseDataset = Dataset<LoadTarget>;
  using BaseDatasetUnqPtr = std::unique_ptr<BaseDataset>;
  using LoadTargetShdPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  EpochShuffleDataset(int64_t random_seed, BaseDatasetUnqPtr&& dataset)
      : base_(std::move(dataset)),
        seed_(random_seed),
        cur_epoch_(0),
        cur_idx_(0),
        index_seq_(dataset->Size()) {
    CHECK(base_->EnableRandomAccess());
    CHECK(base_->EnableGetSize());
    random_engine_.seed(seed_);
    std::iota(index_seq_.begin(), index_seq_.end(), 0);
    std::shuffle(index_seq_.begin(), index_seq_.end(), random_engine_);
  }
  ~EpochShuffleDataset() = default;

  LoadTargetShdPtrVec Next() override {
    LoadTargetShdPtrVec ret;
    LoadTargetShdPtr sample = At(cur_idx_);
    ret.push_back(std::move(sample));
    return ret;
  }

  LoadTargetShdPtr At(int64_t idx) override {
    int64_t epoch = idx / base_->Size();
    int64_t idx_in_epoch = idx % base_->Size();
    CHECK(epoch == cur_epoch_ || epoch == cur_epoch_ + 1);
    if (epoch == cur_epoch_ + 1) {
      std::iota(index_seq_.begin(), index_seq_.end(), 0);
      std::shuffle(index_seq_.begin(), index_seq_.end(), random_engine_);
    }
    int64_t index = index_seq_.at(idx_in_epoch);
    return base_->At(index);
  }

  int64_t Size() override { return base_->Size(); }

  bool EnableRandomAccess() override { return true; }
  bool EnableGetSize() override { return true; }

 private:
  BaseDatasetUnqPtr base_;
  std::mt19937 random_engine_;
  int64_t seed_;
  int64_t cur_epoch_;
  int64_t cur_idx_;
  std::vector<int64_t> index_seq_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_EPOCH_SHUFFLE_DATASET_H_
