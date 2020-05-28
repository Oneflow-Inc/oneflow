#ifndef ONEFLOW_CUSTOMIZED_DATA_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_DATASET_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {
namespace data {

static constexpr int kOneflowDatasetSeed = 524287;

template<typename LoadTarget>
class Dataset {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  Dataset() = default;
  virtual ~Dataset() = default;

  virtual LoadTargetPtrList Next() = 0;
};

template<typename LoadTarget>
class RandomAccessDataset : public Dataset<LoadTarget> {
 public:
  using LoadTargetShdPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  RandomAccessDataset() : cur_idx_(0) {}
  virtual ~RandomAccessDataset() = default;

  virtual LoadTargetShdPtrVec At(int64_t index) const = 0;
  virtual size_t Size() const = 0;

  LoadTargetShdPtrVec Next() final {
    LoadTargetShdPtrVec ret = this->At(cur_idx_);
    cur_idx_ += 1;
    if (cur_idx_ >= this->Size()) { cur_idx_ %= this->Size(); }
    return ret;
  }

 private:
  int64_t cur_idx_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_DATASET_H_
