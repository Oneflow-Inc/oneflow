#include "oneflow/core/job/logical_blob_parallel_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

bool operator==(const LogicalBlobParallelDesc& lhs, const LogicalBlobParallelDesc& rhs) {
  return PbMd().Equivalent(lhs, rhs);
}

bool operator!=(const LogicalBlobParallelDesc& lhs, const LogicalBlobParallelDesc& rhs) {
  return !(lhs == rhs);
}

//  S -> S
//  P -> C
//  C -> P
LogicalBlobParallelDesc GetDualLbpd(const LogicalBlobParallelDesc& lbpd) {
  LogicalBlobParallelDesc ret(lbpd);
  if (lbpd.has_split_parallel()) {
    //  do nothing
  } else if (lbpd.has_broadcast_parallel()) {
    ret.mutable_partial_sum_parallel();
  } else if (lbpd.has_partial_sum_parallel()) {
    ret.mutable_broadcast_parallel();
  } else {
    UNIMPLEMENTED();
  }
  return ret;
}

}  // namespace oneflow
