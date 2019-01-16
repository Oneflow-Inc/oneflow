#include "oneflow/core/job/blob_parallel_desc.h"

namespace oneflow {

BlobParallelDesc& BlobParallelDesc::operator=(const BlobParallelDesc& blob_parallel_desc) {
  CHECK_EQ(model_split_axis_, blob_parallel_desc.model_split_axis_);
  blob_parallel_conf_ = blob_parallel_desc.blob_parallel_conf_;
  return *this;
}

const BlobDataParallel& BlobParallelDesc::data_parallel() const {
  CHECK(blob_parallel_conf_.has_data_parallel());
  return blob_parallel_conf_.data_parallel();
}

const BlobModelParallel& BlobParallelDesc::model_parallel() const {
  CHECK(blob_parallel_conf_.has_model_parallel());
  return blob_parallel_conf_.model_parallel();
}

const BlobGridParallel& BlobParallelDesc::grid_parallel() const {
  CHECK(blob_parallel_conf_.has_grid_parallel());
  return blob_parallel_conf_.grid_parallel();
}

bool operator==(const BlobParallelConf& lhs, const BlobParallelConf& rhs) {
  if (lhs.parallel_type_case() != rhs.parallel_type_case()) { return false; }
  switch (lhs.parallel_type_case()) {
    case BlobParallelConf::ParallelTypeCase::kDataParallel:
      return lhs.data_parallel() == rhs.data_parallel();
    case BlobParallelConf::ParallelTypeCase::kModelParallel:
      return lhs.model_parallel() == rhs.model_parallel();
    case BlobParallelConf::ParallelTypeCase::kGridParallel:
      return lhs.grid_parallel() == rhs.grid_parallel();
    default: UNIMPLEMENTED();
  }
  return true;
}

bool operator==(const BlobDataParallel& lhs, const BlobDataParallel& rhs) {
  return lhs.data_split_num() == rhs.data_split_num() && lhs.clone_num() == rhs.clone_num();
}

bool operator==(const BlobModelParallel& lhs, const BlobModelParallel& rhs) {
  return lhs.clone_num() == rhs.clone_num() && lhs.model_split_num() == rhs.model_split_num();
}

bool operator==(const BlobGridParallel& lhs, const BlobGridParallel& rhs) {
  return lhs.data_split_num() == rhs.data_split_num()
         && lhs.model_split_num() == rhs.model_split_num();
}

}  // namespace oneflow
