#include "oneflow/core/job/blob_parallel_desc.h"

namespace oneflow {

namespace {

int64_t GetParallelNum(const DataBlobParallel& data_blob_parallel) {
  return data_blob_parallel.data_split_num() * data_blob_parallel.clone_num();
}

int64_t GetParallelNum(const ModelBlobParallel& model_blob_parallel) {
  return model_blob_parallel.clone_num() * model_blob_parallel.model_split_num();
}

int64_t GetParallelNum(const GridBlobParallel& grid_blob_parallel) {
  return grid_blob_parallel.data_split_num() * grid_blob_parallel.model_split_num();
}

}  // namespace

BlobParallelDesc& BlobParallelDesc::operator=(const BlobParallelDesc& blob_parallel_desc) {
  CHECK_EQ(model_split_axis_, blob_parallel_desc.model_split_axis_);
  blob_parallel_conf_ = blob_parallel_desc.blob_parallel_conf_;
  return *this;
}

const DataBlobParallel& BlobParallelDesc::data_blob_parallel() const {
  CHECK(blob_parallel_conf_.has_data_blob_parallel());
  return blob_parallel_conf_.data_blob_parallel();
}

const ModelBlobParallel& BlobParallelDesc::model_blob_parallel() const {
  CHECK(blob_parallel_conf_.has_model_blob_parallel());
  return blob_parallel_conf_.model_blob_parallel();
}

const GridBlobParallel& BlobParallelDesc::grid_blob_parallel() const {
  CHECK(blob_parallel_conf_.has_grid_blob_parallel());
  return blob_parallel_conf_.grid_blob_parallel();
}

void BlobParallelDesc::GetDataAxisParallelInfo(bool* is_split, int32_t* axis,
                                               int64_t* axis_parallel_num) const {
  switch (blob_parallel_conf_.parallel_type_case()) {
    case BlobParallelConf::ParallelTypeCase::kDataBlobParallel:
      return GetDataAxisParallelInfo(blob_parallel_conf_.data_blob_parallel(), is_split, axis,
                                     axis_parallel_num);
    case BlobParallelConf::ParallelTypeCase::kModelBlobParallel:
      return GetDataAxisParallelInfo(blob_parallel_conf_.model_blob_parallel(), is_split, axis,
                                     axis_parallel_num);
    case BlobParallelConf::ParallelTypeCase::kGridBlobParallel:
      return GetDataAxisParallelInfo(blob_parallel_conf_.grid_blob_parallel(), is_split, axis,
                                     axis_parallel_num);
    default: UNIMPLEMENTED();
  }
}

void BlobParallelDesc::GetModelAxisParallelInfo(bool* is_split, int32_t* axis,
                                                int64_t* axis_parallel_num) const {
  switch (blob_parallel_conf_.parallel_type_case()) {
    case BlobParallelConf::ParallelTypeCase::kDataBlobParallel:
      return GetModelAxisParallelInfo(blob_parallel_conf_.data_blob_parallel(), is_split, axis,
                                      axis_parallel_num);
    case BlobParallelConf::ParallelTypeCase::kModelBlobParallel:
      return GetModelAxisParallelInfo(blob_parallel_conf_.model_blob_parallel(), is_split, axis,
                                      axis_parallel_num);
    case BlobParallelConf::ParallelTypeCase::kGridBlobParallel:
      return GetModelAxisParallelInfo(blob_parallel_conf_.grid_blob_parallel(), is_split, axis,
                                      axis_parallel_num);
    default: UNIMPLEMENTED();
  }
}

void BlobParallelDesc::GetDataAxisParallelInfo(const DataBlobParallel& data_blob_parallel,
                                               bool* is_split, int32_t* axis,
                                               int64_t* axis_parallel_num) const {
  *is_split = true;
  *axis = 0;
  *axis_parallel_num = data_blob_parallel.data_split_num();
}

void BlobParallelDesc::GetDataAxisParallelInfo(const ModelBlobParallel& model_blob_parallel,
                                               bool* is_split, int32_t* axis,
                                               int64_t* axis_parallel_num) const {
  *is_split = false;
  *axis = -1;
  *axis_parallel_num = model_blob_parallel.clone_num();
}

void BlobParallelDesc::GetDataAxisParallelInfo(const GridBlobParallel& grid_blob_parallel,
                                               bool* is_split, int32_t* axis,
                                               int64_t* axis_parallel_num) const {
  *is_split = true;
  *axis = 0;
  *axis_parallel_num = grid_blob_parallel.data_split_num();
}

void BlobParallelDesc::GetModelAxisParallelInfo(const DataBlobParallel& data_blob_parallel,
                                                bool* is_split, int32_t* axis,
                                                int64_t* axis_parallel_num) const {
  *is_split = false;
  *axis = -1;
  *axis_parallel_num = data_blob_parallel.clone_num();
}

void BlobParallelDesc::GetModelAxisParallelInfo(const ModelBlobParallel& model_blob_parallel,
                                                bool* is_split, int32_t* axis,
                                                int64_t* axis_parallel_num) const {
  *is_split = true;
  CHECK(has_model_split_axis());
  *axis = model_split_axis();
  *axis_parallel_num = model_blob_parallel.model_split_num();
}

void BlobParallelDesc::GetModelAxisParallelInfo(const GridBlobParallel& grid_blob_parallel,
                                                bool* is_split, int32_t* axis,
                                                int64_t* axis_parallel_num) const {
  *is_split = true;
  CHECK(has_model_split_axis());
  *axis = model_split_axis();
  *axis_parallel_num = grid_blob_parallel.model_split_num();
}

int64_t BlobParallelDesc::ParallelNum() const {
  switch (blob_parallel_conf_.parallel_type_case()) {
    case BlobParallelConf::ParallelTypeCase::kDataBlobParallel:
      return GetParallelNum(blob_parallel_conf_.data_blob_parallel());
    case BlobParallelConf::ParallelTypeCase::kModelBlobParallel:
      return GetParallelNum(blob_parallel_conf_.model_blob_parallel());
    case BlobParallelConf::ParallelTypeCase::kGridBlobParallel:
      return GetParallelNum(blob_parallel_conf_.grid_blob_parallel());
    default: UNIMPLEMENTED();
  }
  return true;
}

bool operator==(const BlobParallelConf& lhs, const BlobParallelConf& rhs) {
  if (lhs.parallel_type_case() != rhs.parallel_type_case()) { return false; }
  switch (lhs.parallel_type_case()) {
    case BlobParallelConf::ParallelTypeCase::kDataBlobParallel:
      return lhs.data_blob_parallel() == rhs.data_blob_parallel();
    case BlobParallelConf::ParallelTypeCase::kModelBlobParallel:
      return lhs.model_blob_parallel() == rhs.model_blob_parallel();
    case BlobParallelConf::ParallelTypeCase::kGridBlobParallel:
      return lhs.grid_blob_parallel() == rhs.grid_blob_parallel();
    default: UNIMPLEMENTED();
  }
  return true;
}

bool operator==(const DataBlobParallel& lhs, const DataBlobParallel& rhs) {
  return lhs.data_split_num() == rhs.data_split_num() && lhs.clone_num() == rhs.clone_num();
}

bool operator==(const ModelBlobParallel& lhs, const ModelBlobParallel& rhs) {
  return lhs.clone_num() == rhs.clone_num() && lhs.model_split_num() == rhs.model_split_num();
}

bool operator==(const GridBlobParallel& lhs, const GridBlobParallel& rhs) {
  return lhs.data_split_num() == rhs.data_split_num()
         && lhs.model_split_num() == rhs.model_split_num();
}

}  // namespace oneflow
