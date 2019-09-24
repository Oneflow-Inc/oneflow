#ifndef ONEFLOW_CORE_DATA_DATASET_MANAGER_H_
#define ONEFLOW_CORE_DATA_DATASET_MANAGER_H_

#include "oneflow/core/data/dataset.h"
#include "oneflow/core/data/data.pb.h"

namespace oneflow {
namespace data {

class DatasetManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DatasetManager);
  DatasetManager() = default;
  ~DatasetManager() = default;
  std::shared_ptr<Dataset> Get(DatasetProto::DatasetCatalogCase cat) const;
  std::shared_ptr<Dataset> GetOrCreateDataset(const DatasetProto& proto);

 private:
  HashMap<DatasetProto::DatasetCatalogCase, std::shared_ptr<Dataset>, std::hash<int>>
      dataset_cat2dataset_;
  std::mutex mtx_;
};

inline std::shared_ptr<Dataset> DatasetManager::Get(DatasetProto::DatasetCatalogCase cat) const {
  return dataset_cat2dataset_.at(cat);
}

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

#endif  // ONEFLOW_CORE_DATA_DATASET_MANAGER_H_
