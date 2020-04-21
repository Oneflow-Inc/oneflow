#ifndef ONEFLOW_CUSTOMIZED_DATA_RANDOM_SHUFFLE_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_RANDOM_SHUFFLE_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

template<typename LoadTarget>
class RandomShuffleDataset final : public Dataset<LoadTarget> {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  RandomShuffleDataset(user_op::KernelInitContext* ctx,
                       std::unique_ptr<Dataset<LoadTarget>>&& data_set)
      : loader_(std::move(data_set)) {
    // random
    seed_ = ctx->GetAttr<int64_t>("seed");
    if (seed_ == -1) { seed_ = NewRandomSeed(); }
    std::seed_seq seq({seed_});
    e_ = std::default_random_engine(seq);

    // fill buffer
    initial_buffer_fill_ = ctx->GetAttr<int32_t>("shuffle_buffer_size");
    int32_t remain_cnt = initial_buffer_fill_;
    while (remain_cnt > 0) {
      LoadTargetPtrList sample_list = loader_->Next();
      for (auto& sample_ptr : sample_list) {
        sample_buffer_.push_back(std::move(sample_ptr));
        remain_cnt--;
      }
    }
  }
  ~RandomShuffleDataset() = default;

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret = loader_->Next();
    for (auto& sample_ptr : ret) {
      std::uniform_int_distribution<> dis(0, sample_buffer_.size() - 1);
      int offset = dis(e_);
      std::swap(sample_buffer_[offset], sample_ptr);
    }
    return ret;
  }

  LoadTargetPtr At(int64_t idx) override { return loader_->At(idx); }

  int64_t Size() override { return loader_->Size(); }

  bool EnableRandomAccess() override { return loader_->EnableRandomAccess(); }
  bool EnableGetSize() override { return loader_->EnableGetSize(); }

 private:
  std::unique_ptr<Dataset<LoadTarget>> loader_;
  std::vector<LoadTargetPtr> sample_buffer_;

  int32_t initial_buffer_fill_;

  std::default_random_engine e_;
  int64_t seed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_RANDOM_SHUFFLE_DATASET_H_
