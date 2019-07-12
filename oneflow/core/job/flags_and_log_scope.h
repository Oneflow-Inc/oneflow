#ifndef ONEFLOW_CORE_JOB_FLAGS_AND_LOG_SCOPE_H_
#define ONEFLOW_CORE_JOB_FLAGS_AND_LOG_SCOPE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_set.pb.h"

namespace oneflow {

class FlagsAndLogScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FlagsAndLogScope)
  FlagsAndLogScope(const JobSet& job_set, const char* binary_name);
  ~FlagsAndLogScope();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_FLAGS_AND_LOG_SCOPE_H_
