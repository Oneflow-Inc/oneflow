#ifndef ONEFLOW_CORE_JOB_BLOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_BLOB_PARALLEL_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

bool operator==(const BlobParallelConf& lhs, const BlobParallelConf& rhs);
bool operator==(const BlobDataParallel& lhs, const BlobDataParallel& rhs);
bool operator==(const BlobModelParallel& lhs, const BlobModelParallel& rhs);

class BlobParallelDesc final {
 public:
  BlobParallelDesc() = delete;
  ~BlobParallelDesc() = delete;

  BlobParallelDesc(const BlobParallelDesc&) = default;
  explicit BlobParallelDesc(const BlobParallelConf& blob_parallel_conf)
      : blob_parallel_conf_(blob_parallel_conf) {}

  const BlobDataParallel& data_parallel() const;
  const BlobModelParallel& model_parallel() const;
  int64_t model_split_axis() const;

  bool operator==(const BlobParallelDesc& rhs) const {
    return blob_parallel_conf_ == rhs.blob_parallel_conf_;
  }
  bool operator!=(const BlobParallelDesc& rhs) const { return !(*this == rhs); }

  void set_model_split_axis(int64_t val) { blob_parallel_conf_.set_model_split_axis(val); }
  BlobDataParallel* mut_data_parallel() { return blob_parallel_conf_.mutable_data_parallel(); }
  BlobModelParallel* mut_model_parallel() { return blob_parallel_conf_.mutable_model_parallel(); }

 private:
  BlobParallelConf blob_parallel_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_BLOB_PARALLEL_DESC_H_
