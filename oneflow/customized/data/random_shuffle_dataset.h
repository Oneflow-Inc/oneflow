#ifndef ONEFLOW_CUSTOMIZED_DATA_RANDOM_SHUFFLE_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_RANDOM_SHUFFLE_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

template<typename LoadTarget>
class RandomShuffleDataset final : public Dataset<LoadTarget> {
 public:
  using LoadTargetSharedPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetUniquePtr = std::unique_ptr<LoadTarget>;
  using BatchLoadTargetPtr = std::vector<LoadTargetSharedPtr>;
  RandomShuffleDataset(user_op::KernelInitContext* ctx,
                       std::unique_ptr<Dataset<LoadTarget>>&& data_set)
      : loader_(std::move(data_set)) {
    // random
    seed_ = ctx->GetAttr<int64_t>("seed");
    if (seed_ == -1) { seed_ = NewRandomSeed(); }
    std::seed_seq seq({seed_});
    e_ = std::default_random_engine(seq);

    // empty and buffer
    initial_buffer_fill_ = ctx->GetAttr<int32_t>("initial_fill");
    int32_t batch_size = ctx->GetAttr<int32_t>("batch_size");
    int64_t total_empty_size = initial_buffer_fill_ + 2 * 2 * batch_size;  // maybe 2 * batch_size
    int64_t tensor_init_bytes = ctx->GetAttr<int64_t>("tensor_init_bytes");

    empty_tensor_mgr_.reset(
        new EmptyTensorManager<LoadTarget>(total_empty_size, tensor_init_bytes));
  }
  ~RandomShuffleDataset() = default;

  BatchLoadTargetPtr LoadBatch(int64_t batch_size) override {
    BatchLoadTargetPtr ret;
    for (int i = 0; i < batch_size; ++i) {
      std::uniform_int_distribution<> dis(0, sample_buffer_.size() - 1);
      int offset = dis(e_);
      LoadTargetSharedPtr sample_ptr(
          sample_buffer_.at(offset).release(),
          [this](LoadTarget* sample) { empty_tensor_mgr_->Recycle(sample); });
      ret.push_back(std::move(sample_ptr));
      LoadTargetUniquePtr new_sample = std::move(empty_tensor_mgr_->Get());
      Next(*new_sample);
      sample_buffer_.at(offset) = std::move(new_sample);
    }
    return ret;
  }

  void Next(LoadTarget& tensor) override { loader_->Next(tensor); }
  void At(int64_t idx, LoadTarget& tensor) override { loader_->At(idx, tensor); }

  int64_t Size() override { return loader_->Size(); }

  bool EnableRandomAccess() override { return loader_->EnableRandomAccess(); }
  bool EnableGetSize() override { return loader_->EnableGetSize(); }

 private:
  std::unique_ptr<Dataset<LoadTarget>> loader_;
  std::vector<LoadTargetUniquePtr> sample_buffer_;

  int32_t initial_buffer_fill_;

  std::default_random_engine e_;
  int64_t seed_;

  std::unique_ptr<EmptyTensorManager<LoadTarget>> empty_tensor_mgr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_RANDOM_SHUFFLE_DATASET_H_
