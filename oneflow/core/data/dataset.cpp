#include "oneflow/core/data/dataset.h"
#include "oneflow/core/data/data_sampler.h"

namespace oneflow {
namespace data {

Dataset::Dataset(const DatasetProto& dataset_proto) : sampler_(new DataSampler(this)) {
  dataset_proto_ = &dataset_proto;
  data_seq_.clear();
  Init();
}

}  // namespace data
}  // namespace oneflow
