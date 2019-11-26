#ifndef ONEFLOW_CORE_JOB_INTER_JOB_MEM_SHARING_UTIL_H_
#define ONEFLOW_CORE_JOB_INTER_JOB_MEM_SHARING_UTIL_H_

#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

struct InterJobMemSharingUtil {
  static void MergeMemSharedInterfaceMemBlockBetweenJobs(const std::vector<Job>& jobs, Plan* plan);

  static void MergeMemReusedChunkBetweenUserJobs(const std::vector<Job>& jobs, Plan* plan,
                                                 int64_t user_job_size);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_INTER_JOB_MEM_SHARING_UTIL_H_
