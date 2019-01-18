#include "oneflow/core/job/blob_parallel_desc.h"

namespace oneflow {

namespace {

int64_t GetParallelNum(const BlobDataParallel& blob_data_parallel) {
  return blob_data_parallel.data_split_num() * blob_data_parallel.clone_num();
}

int64_t GetParallelNum(const BlobModelParallel& blob_model_parallel) {
  return blob_model_parallel.clone_num() * blob_model_parallel.model_split_num();
}

int64_t GetParallelNum(const BlobGridParallel& blob_grid_parallel) {
  return blob_grid_parallel.data_split_num() * blob_grid_parallel.model_split_num();
}

}  // namespace

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

void BlobParallelDesc::GetDataAxisParallelInfo(bool* is_split, int32_t* axis,
                                               int64_t* axis_parallel_num) const {
  switch (blob_parallel_conf_.parallel_type_case()) {
    case BlobParallelConf::ParallelTypeCase::kDataParallel:
      return GetDataAxisParallelInfo(blob_parallel_conf_.data_parallel(), is_split, axis,
                                     axis_parallel_num);
    case BlobParallelConf::ParallelTypeCase::kModelParallel:
      return GetDataAxisParallelInfo(blob_parallel_conf_.model_parallel(), is_split, axis,
                                     axis_parallel_num);
    case BlobParallelConf::ParallelTypeCase::kGridParallel:
      return GetDataAxisParallelInfo(blob_parallel_conf_.grid_parallel(), is_split, axis,
                                     axis_parallel_num);
    default: UNIMPLEMENTED();
  }
}

void BlobParallelDesc::GetModelAxisParallelInfo(bool* is_split, int32_t* axis,
                                                int64_t* axis_parallel_num) const {
  switch (blob_parallel_conf_.parallel_type_case()) {
    case BlobParallelConf::ParallelTypeCase::kDataParallel:
      return GetModelAxisParallelInfo(blob_parallel_conf_.data_parallel(), is_split, axis,
                                      axis_parallel_num);
    case BlobParallelConf::ParallelTypeCase::kModelParallel:
      return GetModelAxisParallelInfo(blob_parallel_conf_.model_parallel(), is_split, axis,
                                      axis_parallel_num);
    case BlobParallelConf::ParallelTypeCase::kGridParallel:
      return GetModelAxisParallelInfo(blob_parallel_conf_.grid_parallel(), is_split, axis,
                                      axis_parallel_num);
    default: UNIMPLEMENTED();
  }
}

void BlobParallelDesc::GetDataAxisParallelInfo(const BlobDataParallel& blob_data_parallel,
                                               bool* is_split, int32_t* axis,
                                               int64_t* axis_parallel_num) const {
  *is_split = true;
  *axis = 0;
  *axis_parallel_num = blob_data_parallel.data_split_num();
}

void BlobParallelDesc::GetDataAxisParallelInfo(const BlobModelParallel& blob_model_parallel,
                                               bool* is_split, int32_t* axis,
                                               int64_t* axis_parallel_num) const {
  *is_split = false;
  *axis = -1;
  *axis_parallel_num = blob_model_parallel.clone_num();
}

void BlobParallelDesc::GetDataAxisParallelInfo(const BlobGridParallel& blob_grid_parallel,
                                               bool* is_split, int32_t* axis,
                                               int64_t* axis_parallel_num) const {
  *is_split = true;
  *axis = 0;
  *axis_parallel_num = blob_grid_parallel.data_split_num();
}

void BlobParallelDesc::GetModelAxisParallelInfo(const BlobDataParallel& blob_data_parallel,
                                                bool* is_split, int32_t* axis,
                                                int64_t* axis_parallel_num) const {
  *is_split = false;
  *axis = -1;
  *axis_parallel_num = blob_data_parallel.clone_num();
}

void BlobParallelDesc::GetModelAxisParallelInfo(const BlobModelParallel& blob_model_parallel,
                                                bool* is_split, int32_t* axis,
                                                int64_t* axis_parallel_num) const {
  *is_split = true;
  CHECK(has_model_split_axis());
  *axis = model_split_axis();
  *axis_parallel_num = blob_model_parallel.model_split_num();
}

void BlobParallelDesc::GetModelAxisParallelInfo(const BlobGridParallel& blob_grid_parallel,
                                                bool* is_split, int32_t* axis,
                                                int64_t* axis_parallel_num) const {
  *is_split = true;
  CHECK(has_model_split_axis());
  *axis = model_split_axis();
  *axis_parallel_num = blob_grid_parallel.model_split_num();
}

int64_t BlobParallelDesc::ParallelNum() const {
  switch (blob_parallel_conf_.parallel_type_case()) {
    case BlobParallelConf::ParallelTypeCase::kDataParallel:
      return GetParallelNum(blob_parallel_conf_.data_parallel());
    case BlobParallelConf::ParallelTypeCase::kModelParallel:
      return GetParallelNum(blob_parallel_conf_.model_parallel());
    case BlobParallelConf::ParallelTypeCase::kGridParallel:
      return GetParallelNum(blob_parallel_conf_.grid_parallel());
    default: UNIMPLEMENTED();
  }
  return true;
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
