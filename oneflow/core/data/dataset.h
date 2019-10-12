#ifndef ONEFLOW_CORE_DATA_DATASET_H_
#define ONEFLOW_CORE_DATA_DATASET_H_

#include "oneflow/core/data/data_instance.h"
#include "oneflow/core/data/data_sampler.h"
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {
namespace data {

class Dataset {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Dataset);
  Dataset() = delete;
  explicit Dataset(const DatasetProto& proto);
  virtual ~Dataset() = default;

  virtual size_t Size() const = 0;
  virtual void GetData(int64_t idx, DataInstance* data) const = 0;
  virtual int64_t GetGroupId(int64_t idx) const { UNIMPLEMENTED(); }
  void SubmitSamplerContext(DataSamplerContext* ctx) { sampler_->SubmitContext(ctx); }
  std::vector<int64_t> FetchBatchIndexSequence(DataSamplerContext* ctx, size_t batch_size) {
    return sampler_->FetchBatchIndexSequence(ctx, batch_size);
  }

  const DatasetProto& dataset_proto() const { return *dataset_proto_; }
  std::unique_ptr<DataSampler>& sampler() { return sampler_; }

 private:
  const DatasetProto* dataset_proto_;
  std::unique_ptr<DataSampler> sampler_;
};

#define DATASET_CASE_SEQ                                            \
  OF_PP_MAKE_TUPLE_SEQ(DatasetProto::DatasetCatalogCase::kImageNet) \
  OF_PP_MAKE_TUPLE_SEQ(DatasetProto::DatasetCatalogCase::kCoco)

#define REGISTER_DATASET_CREATOR(k, f) REGISTER_CLASS_CREATOR(k, Dataset, f, const DatasetProto&)

#define MAKE_DATASET_CREATOR_ENTRY(dataset_class, dataset_case) \
  {dataset_case, [](const DatasetProto& proto) -> Dataset* { return new dataset_class(proto); }},

#define REGISTER_DATASET(dataset_case, dataset_derived_class)                                     \
  namespace {                                                                                     \
                                                                                                  \
  Dataset* OF_PP_CAT(CreateDataset, __LINE__)(const DatasetProto& proto) {                        \
    static const HashMap<DatasetProto::DatasetCatalogCase,                                        \
                         std::function<Dataset*(const DatasetProto& proto)>, std::hash<int>>      \
        creators = {OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_DATASET_CREATOR_ENTRY,                  \
                                                     (dataset_derived_class), DATASET_CASE_SEQ)}; \
    return creators.at(proto.dataset_catalog_case())(proto);                                      \
  }                                                                                               \
                                                                                                  \
  REGISTER_DATASET_CREATOR(dataset_case, OF_PP_CAT(CreateDataset, __LINE__));                     \
  }

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATA_DATASET_H_
