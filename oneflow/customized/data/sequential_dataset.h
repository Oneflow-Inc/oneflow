#ifndef ONEFLOW_CUSTOMIZED_DATA_SEQUENTIAL_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_SEQUENTIAL_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

template<typename LoadTarget>
class SequentialDataset final : public Dataset<LoadTarget> {
 public:
  using LoadTargetSharedPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetUniquePtr = std::unique_ptr<LoadTarget>;
  using BatchLoadTargetPtr = std::vector<LoadTargetSharedPtr>;
  SequentialDataset(user_op::KernelInitContext* ctx,
                    std::unique_ptr<Dataset<LoadTarget>>&& data_set)
      : loader_(std::move(data_set)) {
    int32_t batch_size = ctx->GetAttr<int32_t>("batch_size");
    int64_t total_empty_size = 2 * 2 * batch_size;  // maybe 2 * batch_size
    int64_t tensor_init_bytes = ctx->GetAttr<int64_t>("tensor_init_bytes");
    empty_tensor_mgr_.reset(
        new EmptyTensorManager<LoadTarget>(total_empty_size, tensor_init_bytes));
  }
  ~SequentialDataset() = default;

  BatchLoadTargetPtr LoadBatch(int64_t batch_size) override {
    BatchLoadTargetPtr ret;
    for (int i = 0; i < batch_size; ++i) {
      LoadTargetSharedPtr sample_ptr(
          empty_tensor_mgr_->Get().release(),
          [this](LoadTarget* sample) { empty_tensor_mgr_->Recycle(sample); });
      Next(*sample_ptr);
      ret.push_back(std::move(sample_ptr));
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
  std::unique_ptr<EmptyTensorManager<LoadTarget>> empty_tensor_mgr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_SEQUENTIAL_DATASET_H_
