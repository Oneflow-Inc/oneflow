#ifndef ONEFLOW_CORE_JOB_INTER_JOB_MEM_SHARING_UTIL_H_
#define ONEFLOW_CORE_JOB_INTER_JOB_MEM_SHARING_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

struct InterJobMemSharingUtil {
  static void BindInterfaceMemBlockId(const std::vector<Job>& jobs, std::vector<Plan>* sub_plans);

  static void MergeMemBlockBetweenSubPlans(const std::vector<Job>& jobs,
                                           std::vector<Plan>* sub_plans);

  static std::vector<HashSet<int64_t>> GetMutualExclusionJobGroups(const std::vector<Job>& jobs);

  static void MergeAndCleanChunkBlock(Plan* plan,
                                      const std::vector<HashSet<int64_t>>& reuse_mem_job_groups);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_INTER_JOB_MEM_SHARING_UTIL_H_
