#include "oneflow/core/data/dataset.h"

namespace oneflow {
namespace data {

Dataset::Dataset(const DatasetProto& dataset_proto)
    : dataset_proto_(&dataset_proto), sampler_(new DataSampler(this)) {}

}  // namespace data
}  // namespace oneflow
