#ifndef ONEFLOW_CORE_DATA_DATASET_MANAGER_H_
#define ONEFLOW_CORE_DATA_DATASET_MANAGER_H_

#include "oneflow/core/data/dataset.h"

namespace oneflow {
namespace data {

class DatasetManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DatasetManager);
  DatasetManager() = default;
  ~DatasetManager() = default;
  std::shared_ptr<Dataset> Get(DatasetProto::DatasetCatalogCase cat) const {
    return dataset_cat2dataset_.at(cat);
  }
  std::shared_ptr<Dataset> GetOrCreateDataset(const DatasetProto& proto);

 private:
  HashMap<DatasetProto::DatasetCatalogCase, std::shared_ptr<Dataset>, std::hash<int>>
      dataset_cat2dataset_;
  std::mutex mtx_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATA_DATASET_MANAGER_H_
