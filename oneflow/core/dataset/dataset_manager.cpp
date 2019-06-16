#include "oneflow/core/dataset/dataset_manager.h"

namespace oneflow {

DatasetManager::DatasetManager(const JobDesc* job_desc) {
  for (const auto& pair : job_desc->dataset_cluster().dataset_map()) {
    const DatasetProto& dataset_proto = pair.second;
    Dataset* dataset = NewObj<Dataset>(dataset_proto.dataset_catalog_case(), dataset_proto);
    dataset->Init(dataset_proto);
    int64_t total_data_num =
        job_desc->IsTrain() ? job_desc->TotalBatchNum() * job_desc->BatchSize() : dataset->Size();
    dataset->GenDataSequence(total_data_num);
    CHECK(dataset_map_.emplace(pair.first, std::shared_ptr<Dataset>(dataset)).second);
  }
}

}  // namespace oneflow
