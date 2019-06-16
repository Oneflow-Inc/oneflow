#ifndef ONEFLOW_CORE_DATASET_DATASET_MANAGER_H_
#define ONEFLOW_CORE_DATASET_DATASET_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/dataset/dataset.h"
#include "oneflow/core/job/job_desc.h"
#include <memory>

namespace oneflow {

class DatasetManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DatasetManager);
  DatasetManager() = delete;
  ~DatasetManager() = default;
  DatasetManager(const JobDesc* job_desc);
  std::shared_ptr<Dataset> At(const std::string& name) const { return dataset_map_.at(name); }

 private:
  HashMap<std::string, std::shared_ptr<Dataset>> dataset_map_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATASET_DATASET_MANAGER_H_
