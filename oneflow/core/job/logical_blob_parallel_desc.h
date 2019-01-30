#ifndef ONEFLOW_CORE_JOB_LOGICAL_BLOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_LOGICAL_BLOB_PARALLEL_DESC_H_

#include "oneflow/core/job/logical_blob_parallel_conf.pb.h"

namespace oneflow {

class LogicalBlobParallelDesc final {
 public:
  LogicalBlobParallelDesc() : parallel_num_(0) {}
  ~LogicalBlobParallelDesc() = default;

  // Getters;
  int64_t parallel_num() const;
  const SplitParallel& split_parallel() const { return lb_parallel_conf_.split_parallel(); }
  bool has_split_parallel() const { return lb_parallel_conf_.has_split_parallel(); }
  bool has_clone_parallel() const { return lb_parallel_conf_.has_clone_parallel(); }
  bool has_partial_sum_parallel() const { return lb_parallel_conf_.has_partial_sum_parallel(); }

  // Setters
  void set_parallel_num(int64_t val) { parallel_num_ = val; }
  SplitParallel* mutable_split_parallel() { return lb_parallel_conf_.mutable_split_parallel(); }
  CloneParallel* mutable_clone_parallel() { return lb_parallel_conf_.mutable_clone_parallel(); }
  PartialSumParallel* mutable_partial_sum_parallel() {
    return lb_parallel_conf_.mutable_partial_sum_parallel();
  }

  bool operator==(const LogicalBlobParallelDesc& rhs) const;
  bool operator!=(const LogicalBlobParallelDesc& rhs) const { return !(*this == rhs); }

 private:
  int64_t parallel_num_;
  LogicalBlobParallelConf lb_parallel_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_LOGICAL_BLOB_PARALLEL_DESC_H_
