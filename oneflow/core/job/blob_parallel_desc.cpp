#include "oneflow/core/job/blob_parallel_desc.h"

namespace oneflow {

const BlobDataParallel& BlobParallelDesc::data_parallel() const {
  CHECK(blob_parallel_conf_.has_data_parallel());
  return blob_parallel_conf_.data_parallel();
}

const BlobModelParallel& BlobParallelDesc::model_parallel() const {
  CHECK(blob_parallel_conf_.has_model_parallel());
  return blob_parallel_conf_.model_parallel();
}

int64_t BlobParallelDesc::model_split_axis() const {
  CHECK(blob_parallel_conf_.has_model_split_axis());
  return blob_parallel_conf_.model_split_axis();
}

bool operator==(const BlobParallelConf& lhs, const BlobParallelConf& rhs) {
  if (lhs.has_model_split_axis() != rhs.has_model_split_axis()) { return false; }
  if (lhs.model_split_axis() != rhs.model_split_axis()) { return false; }
  if (lhs.parallel_type_case() != rhs.parallel_type_case()) { return false; }
  switch (lhs.parallel_type_case()) {
    case BlobParallelConf::ParallelTypeCase::kDataParallel:
      return lhs.data_parallel() == rhs.data_parallel();
    case BlobParallelConf::ParallelTypeCase::kModelParallel:
      return lhs.model_parallel() == rhs.model_parallel();
    case BlobParallelConf::ParallelTypeCase::kGridParallel: TODO();
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

}  // namespace oneflow
