#ifndef ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_
#define ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/graph/plan_task_graph.h"

namespace oneflow {

struct IntraJobMemSharingUtil {
  static void InferMemBlockId4MemReusedRegst(Plan* plan, const PlanTaskGraph& plan_task_graph);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_
