#ifndef ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
#define ONEFLOW_CORE_JOB_SBP_PARALLEL_H_

#include "oneflow/core/job/sbp_parallel.pb.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs);
bool operator!=(const SbpParallel& lhs, const SbpParallel& rhs);
bool operator==(const SbpSignature& lhs, const SbpSignature& rhs);
bool operator!=(const SbpSignature& lhs, const SbpSignature& rhs);

SbpParallel GetDualSbpParallel(const SbpParallel&);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::SbpSignature> {
  size_t operator()(const oneflow::SbpSignature& signature) const {
    std::string serialized_string;
    signature.SerializeToString(&serialized_string);
    return std::hash<std::string>()(serialized_string);
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
