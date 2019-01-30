#include "oneflow/core/job/logical_blob_parallel_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

int64_t LogicalBlobParallelDesc::parallel_num() const {
  CHECK_GT(parallel_num_, 0);
  return parallel_num_;
}

//  S -> S
//  P -> C
//  C -> P
LogicalBlobParallelDesc LogicalBlobParallelDesc::DualLbpd() const {
  LogicalBlobParallelDesc ret(*this);
  if (has_split_parallel()) {
    //  do nothing
  } else if (has_clone_parallel()) {
    ret.mutable_partial_sum_parallel();
  } else if (has_partial_sum_parallel()) {
    ret.mutable_clone_parallel();
  } else {
    UNIMPLEMENTED();
  }
  return ret;
}

bool LogicalBlobParallelDesc::operator==(const LogicalBlobParallelDesc& rhs) const {
  PbMd message_diff;
  return parallel_num_ == rhs.parallel_num_
         && message_diff.Equivalent(lb_parallel_conf_, rhs.lb_parallel_conf_);
}

}  // namespace oneflow
