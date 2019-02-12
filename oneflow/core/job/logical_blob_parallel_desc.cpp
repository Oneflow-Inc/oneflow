#include "oneflow/core/job/logical_blob_parallel_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs) {
  return PbMd().Equivalent(lhs, rhs);
}

bool operator!=(const SbpParallel& lhs, const SbpParallel& rhs) { return !(lhs == rhs); }

//  S -> S
//  P -> C
//  C -> P
SbpParallel GetDualSbpParallel(const SbpParallel& sbp_parallel) {
  SbpParallel ret(sbp_parallel);
  if (sbp_parallel.has_split_parallel()) {
    //  do nothing
  } else if (sbp_parallel.has_broadcast_parallel()) {
    ret.mutable_partial_sum_parallel();
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    ret.mutable_broadcast_parallel();
  } else {
    UNIMPLEMENTED();
  }
  return ret;
}

}  // namespace oneflow
