#ifndef ONEFLOW_CUSTOMIZED_DATA_BATCH_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_BATCH_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

template<typename LoadTarget>
class BatchDataset final : public Dataset<LoadTarget> {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  BatchDataset(int32_t batch_size, std::unique_ptr<Dataset<LoadTarget>>&& data_set)
      : batch_size_(batch_size), loader_(std::move(data_set)) {}
  ~BatchDataset() = default;

  LoadTargetPtrList Next() override {
    int32_t remain_cnt = batch_size_;
    LoadTargetPtrList ret;
    while (remain_cnt > 0) {
      LoadTargetPtrList tmp = loader_->Next();
      for (auto& sample_ptr : tmp) {
        ret.push_back(std::move(sample_ptr));
        remain_cnt--;
      }
    }
    return ret;
  }

 private:
  int32_t batch_size_;
  std::unique_ptr<Dataset<LoadTarget>> loader_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_BATCH_DATASET_H_
