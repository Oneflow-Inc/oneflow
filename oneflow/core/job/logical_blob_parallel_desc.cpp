#include "oneflow/core/job/logical_blob_parallel_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

int64_t LogicalBlobParallelDesc::parallel_num() const {
  CHECK_GT(parallel_num_, 0);
  return parallel_num_;
}

bool LogicalBlobParallelDesc::operator==(const LogicalBlobParallelDesc& rhs) const {
  PbMd message_diff;
  return parallel_num_ == rhs.parallel_num_
         && message_diff.Equivalent(lb_parallel_conf_, rhs.lb_parallel_conf_);
}

}  // namespace oneflow
