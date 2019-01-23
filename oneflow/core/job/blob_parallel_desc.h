#ifndef ONEFLOW_CORE_JOB_BLOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_BLOB_PARALLEL_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

bool operator==(const BlobParallelConf& lhs, const BlobParallelConf& rhs);
bool operator==(const DataBlobParallel& lhs, const DataBlobParallel& rhs);
bool operator==(const ModelBlobParallel& lhs, const ModelBlobParallel& rhs);
bool operator==(const GridBlobParallel& lhs, const GridBlobParallel& rhs);

class BlobParallelDesc final {
 public:
  BlobParallelDesc() = delete;
  ~BlobParallelDesc() = default;

  explicit BlobParallelDesc(const BlobParallelDesc& blob_parallel_desc) = default;
  explicit BlobParallelDesc(int64_t model_split_axis) : model_split_axis_(model_split_axis) {}
  void CopyBlobParallelConf(const BlobParallelDesc& blob_parallel_desc);

  // Getters
  // for data input blob
  const DataBlobParallel& data_blob_parallel() const;
  // for model input blob
  const ModelBlobParallel& model_blob_parallel() const;
  // for output blob or element-wise op's input blob in some cases.
  const GridBlobParallel& grid_blob_parallel() const;
  const BlobParallelConf& blob_parallel_conf() const { return blob_parallel_conf_; }
  void GetDataAxisParallelInfo(bool* is_split, int32_t* axis, int64_t* axis_parallel_num) const;
  void GetModelAxisParallelInfo(bool* is_split, int32_t* axis, int64_t* axis_parallel_num) const;
  int64_t ParallelNum() const;
  int64_t model_split_axis() const { return model_split_axis_; }
  bool has_data_blob_parallel() const { return blob_parallel_conf_.has_data_blob_parallel(); }
  bool has_model_blob_parallel() const { return blob_parallel_conf_.has_model_blob_parallel(); }
  bool has_grid_blob_parallel() const { return blob_parallel_conf_.has_grid_blob_parallel(); }
  bool has_model_split_axis() const { return model_split_axis_ != -1; }

  bool operator==(const BlobParallelDesc& rhs) const {
    return blob_parallel_conf_ == rhs.blob_parallel_conf_;
  }
  bool operator!=(const BlobParallelDesc& rhs) const { return !(*this == rhs); }

  // Setters
  DataBlobParallel* mut_data_blob_parallel() {
    return blob_parallel_conf_.mutable_data_blob_parallel();
  }
  ModelBlobParallel* mut_model_blob_parallel() {
    return blob_parallel_conf_.mutable_model_blob_parallel();
  }
  GridBlobParallel* mut_grid_blob_parallel() {
    return blob_parallel_conf_.mutable_grid_blob_parallel();
  }

 private:
  void GetDataAxisParallelInfo(const DataBlobParallel& data_blob_parallel, bool* is_split,
                               int32_t* axis, int64_t* axis_parallel_num) const;
  void GetDataAxisParallelInfo(const ModelBlobParallel& model_blob_parallel, bool* is_split,
                               int32_t* axis, int64_t* axis_parallel_num) const;
  void GetDataAxisParallelInfo(const GridBlobParallel& grid_blob_parallel, bool* is_split,
                               int32_t* axis, int64_t* axis_parallel_num) const;
  void GetModelAxisParallelInfo(const DataBlobParallel& data_blob_parallel, bool* is_split,
                                int32_t* axis, int64_t* axis_parallel_num) const;
  void GetModelAxisParallelInfo(const ModelBlobParallel& model_blob_parallel, bool* is_split,
                                int32_t* axis, int64_t* axis_parallel_num) const;
  void GetModelAxisParallelInfo(const GridBlobParallel& grid_blob_parallel, bool* is_split,
                                int32_t* axis, int64_t* axis_parallel_num) const;

  const int64_t model_split_axis_;
  BlobParallelConf blob_parallel_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_BLOB_PARALLEL_DESC_H_
