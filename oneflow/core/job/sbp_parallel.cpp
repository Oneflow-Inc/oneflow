#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs) { return PbMd().Equals(lhs, rhs); }

bool operator!=(const SbpParallel& lhs, const SbpParallel& rhs) { return !(lhs == rhs); }

bool operator==(const SbpSignature& lhs, const SbpSignature& rhs) {
  return PbMd().Equals(lhs, rhs);
}

bool operator!=(const SbpSignature& lhs, const SbpSignature& rhs) { return !(lhs == rhs); }

//  S -> S
//  P -> B
//  B -> P
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

bool IsSbpSignatureContaining(const SbpSignature& bigger, const SbpSignature& smaller) {
  auto& bn2sbp = bigger.bn_in_op2sbp_parallel();
  for (const auto& pair : smaller.bn_in_op2sbp_parallel()) {
    CHECK(bn2sbp.find(pair.first) != bn2sbp.end());
    if (bn2sbp.at(pair.first) != pair.second) { return false; }
  }
  return true;
}

}  // namespace oneflow
