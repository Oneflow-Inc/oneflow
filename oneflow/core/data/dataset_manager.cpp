#include "oneflow/core/data/dataset_manager.h"

namespace oneflow {
namespace data {

std::shared_ptr<Dataset> DatasetManager::GetOrCreateDataset(const DatasetProto& proto) {
  std::unique_lock<std::mutex> lck_(mtx_);
  if (dataset_map_.find(proto.name()) == dataset_map_.end()) {
    Dataset* dataset = NewObj<Dataset>(proto.dataset_catalog_case(), proto);
    std::shared_ptr<Dataset> dataset_ptr;
    dataset_ptr.reset(dataset);
    dataset_map_.emplace(proto.name(), dataset_ptr);
  }
  return dataset_map_.at(proto.name());
}

}  // namespace data
}  // namespace oneflow
