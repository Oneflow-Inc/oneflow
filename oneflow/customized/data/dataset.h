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

class Sampler {
 public:
  Sampler() = default;
  virtual ~Sampler() = default;

  virtual int64_t Next() = 0;
};

template<typename LoadTarget>
class RandomAccessDataset : public Dataset<LoadTarget> {
 public:
  using LoadTargetShdPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  RandomAccessDataset() = default;
  virtual ~RandomAccessDataset() = default;

  virtual LoadTargetShdPtr At(int64_t index) const = 0;
  virtual size_t Size() const = 0;

  virtual LoadTargetShdPtrVec Next() override {
    LoadTargetShdPtrVec ret;
    int64_t index = sampler_->Next();
    ret.push_back(std::move(this->At(index)));
    return ret;
  }

  void ResetSampler(std::unique_ptr<Sampler>&& sampler) { sampler_ = std::move(sampler); }
  // Sampler* GetSampler() {
  //   return sampler_.get();
  // }

 private:
  std::unique_ptr<Sampler> sampler_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_DATASET_H_
