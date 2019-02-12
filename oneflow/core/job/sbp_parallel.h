#ifndef ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
#define ONEFLOW_CORE_JOB_SBP_PARALLEL_H_

#include "oneflow/core/job/sbp_parallel.pb.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs);
bool operator!=(const SbpParallel& lhs, const SbpParallel& rhs);
SbpParallel GetDualSbpParallel(const SbpParallel&);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
