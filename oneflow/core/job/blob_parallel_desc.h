#ifndef ONEFLOW_CORE_JOB_BLOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_BLOB_PARALLEL_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

bool operator==(const BlobParallelConf& lhs, const BlobParallelConf& rhs);
bool operator==(const BlobDataParallel& lhs, const BlobDataParallel& rhs);
bool operator==(const BlobModelParallel& lhs, const BlobModelParallel& rhs);
bool operator==(const BlobGridParallel& lhs, const BlobGridParallel& rhs);

class BlobParallelDesc final {
 public:
  BlobParallelDesc() = delete;
  ~BlobParallelDesc() = default;

  explicit BlobParallelDesc(const BlobParallelDesc& blob_parallel_desc) = default;
  explicit BlobParallelDesc(int64_t model_split_axis) : model_split_axis_(model_split_axis) {}
  BlobParallelDesc& operator=(const BlobParallelDesc& blob_parallel_desc);

  // Getters
  // for data input blob
  const BlobDataParallel& data_parallel() const;
  // for model input blob
  const BlobModelParallel& model_parallel() const;
  // for output blob or element-wise op's input blob in some cases.
  const BlobGridParallel& grid_parallel() const;
  const BlobParallelConf& blob_parallel_conf() const { return blob_parallel_conf_; }
  void GetDataAxisParallelInfo(bool* is_split, int32_t* axis, int64_t* axis_parallel_num) const;
  void GetModelAxisParallelInfo(bool* is_split, int32_t* axis, int64_t* axis_parallel_num) const;
  int64_t ParallelNum() const;
  int64_t model_split_axis() const { return model_split_axis_; }
  bool has_data_parallel() const { return blob_parallel_conf_.has_data_parallel(); }
  bool has_model_parallel() const { return blob_parallel_conf_.has_model_parallel(); }
  bool has_grid_parallel() const { return blob_parallel_conf_.has_grid_parallel(); }
  bool has_model_split_axis() const { return model_split_axis_ != -1; }

  bool operator==(const BlobParallelDesc& rhs) const {
    return blob_parallel_conf_ == rhs.blob_parallel_conf_;
  }
  bool operator!=(const BlobParallelDesc& rhs) const { return !(*this == rhs); }

  // Setters
  BlobDataParallel* mut_data_parallel() { return blob_parallel_conf_.mutable_data_parallel(); }
  BlobModelParallel* mut_model_parallel() { return blob_parallel_conf_.mutable_model_parallel(); }
  BlobGridParallel* mut_grid_parallel() { return blob_parallel_conf_.mutable_grid_parallel(); }

 private:
  void GetDataAxisParallelInfo(const BlobDataParallel& blob_data_parallel, bool* is_split,
                               int32_t* axis, int64_t* axis_parallel_num) const;
  void GetDataAxisParallelInfo(const BlobModelParallel& blob_model_parallel, bool* is_split,
                               int32_t* axis, int64_t* axis_parallel_num) const;
  void GetDataAxisParallelInfo(const BlobGridParallel& blob_grid_parallel, bool* is_split,
                               int32_t* axis, int64_t* axis_parallel_num) const;
  void GetModelAxisParallelInfo(const BlobDataParallel& blob_data_parallel, bool* is_split,
                                int32_t* axis, int64_t* axis_parallel_num) const;
  void GetModelAxisParallelInfo(const BlobModelParallel& blob_model_parallel, bool* is_split,
                                int32_t* axis, int64_t* axis_parallel_num) const;
  void GetModelAxisParallelInfo(const BlobGridParallel& blob_grid_parallel, bool* is_split,
                                int32_t* axis, int64_t* axis_parallel_num) const;

  const int64_t model_split_axis_;
  BlobParallelConf blob_parallel_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_BLOB_PARALLEL_DESC_H_
