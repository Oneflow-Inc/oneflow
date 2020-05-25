#ifndef ONEFLOW_CUSTOMIZED_DATA_BATCH_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_BATCH_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
class BatchDataset final : public Dataset<LoadTarget> {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  BatchDataset(int32_t batch_size, std::unique_ptr<Dataset<LoadTarget>>&& data_set)
      : batch_size_(batch_size), loader_(std::move(data_set)) {}
  ~BatchDataset() = default;

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret;
    ret.reserve(batch_size_);
    for (int32_t i = 0; i < batch_size_; ++i) {
      LoadTargetPtrList tmp = loader_->Next();
      CHECK_EQ(tmp.size(), 1);
      ret.push_back(std::move(tmp.at(0)));
    }
    return ret;
  }

 private:
  int32_t batch_size_;
  std::unique_ptr<Dataset<LoadTarget>> loader_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_BATCH_DATASET_H_
