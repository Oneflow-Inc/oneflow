#include "oneflow/core/data/dataset_manager.h"

namespace oneflow {
namespace data {

std::shared_ptr<Dataset> DatasetManager::GetOrCreateDataset(const DatasetProto& proto) {
  std::unique_lock<std::mutex> lck_(mtx_);
  if (dataset_cat2dataset_.find(proto.dataset_catalog_case()) == dataset_cat2dataset_.end()) {
    Dataset* dataset = NewObj<Dataset>(proto.dataset_catalog_case(), proto);
    std::shared_ptr<Dataset> dataset_ptr;
    dataset_ptr.reset(dataset);
    dataset_cat2dataset_.emplace(proto.dataset_catalog_case(), dataset_ptr);
  }
  return dataset_cat2dataset_.at(proto.dataset_catalog_case());
}

}  // namespace data
}  // namespace oneflow
