#ifndef ONEFLOW_CORE_DATA_DATASET_H_
#define ONEFLOW_CORE_DATA_DATASET_H_

#include "oneflow/core/data/data_instance.h"
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {
namespace data {

class DataSampler;

class Dataset {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Dataset);
  Dataset() = delete;
  explicit Dataset(const DatasetProto& proto);
  virtual ~Dataset() = default;

  virtual void Init() {}
  virtual size_t Size() const = 0;
  virtual void GetData(int64_t idx, DataInstance* data) const = 0;
  virtual int64_t GetGroupId(int64_t idx) const { UNIMPLEMENTED(); }
  DataSampler* GetSampler() { return sampler_.get(); }

  const DatasetProto& dataset_proto() const { return *dataset_proto_; }
  std::unique_ptr<DataSampler>& sampler() { return sampler_; }

 private:
  const DatasetProto* dataset_proto_;
  std::unique_ptr<DataSampler> sampler_;

  std::vector<int64_t> data_seq_;
};

}  // namespace data
}  // namespace oneflow

#define REGISTER_DATASET(k, DerivedDatasetClass) \
  REGISTER_CLASS_WITH_ARGS(k, Dataset, DerivedDatasetClass, const DatasetProto&)
#define REGISTER_DATASET_CREATOR(k, f) REGISTER_CLASS_CREATOR(k, Dataset, f, const DatasetProto&)

#endif  // ONEFLOW_CORE_DATA_DATASET_H_
