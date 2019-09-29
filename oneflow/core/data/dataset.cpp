#include "oneflow/core/data/dataset.h"

namespace oneflow {
namespace data {

Dataset::Dataset(const DatasetProto& dataset_proto) : sampler_(new DataSampler(this)) {
  dataset_proto_ = &dataset_proto;
}

}  // namespace data
}  // namespace oneflow
