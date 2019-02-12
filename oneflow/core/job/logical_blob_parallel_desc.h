#ifndef ONEFLOW_CORE_JOB_LOGICAL_BLOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_LOGICAL_BLOB_PARALLEL_DESC_H_

#include "oneflow/core/job/logical_blob_parallel_desc.pb.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs);
bool operator!=(const SbpParallel& lhs, const SbpParallel& rhs);
SbpParallel GetDualLbpd(const SbpParallel&);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_LOGICAL_BLOB_PARALLEL_DESC_H_
