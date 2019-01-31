#ifndef ONEFLOW_CORE_JOB_LOGICAL_BLOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_LOGICAL_BLOB_PARALLEL_DESC_H_

#include "oneflow/core/job/logical_blob_parallel_desc.pb.h"

namespace oneflow {

bool operator==(const LogicalBlobParallelDesc& lhs, const LogicalBlobParallelDesc& rhs);
bool operator!=(const LogicalBlobParallelDesc& lhs, const LogicalBlobParallelDesc& rhs);
LogicalBlobParallelDesc GetDualLbpd(const LogicalBlobParallelDesc&);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_LOGICAL_BLOB_PARALLEL_DESC_H_
