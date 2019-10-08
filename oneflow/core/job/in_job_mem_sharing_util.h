#ifndef ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_
#define ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_

#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

struct InJobMemSharingUtil {
  static void InferMemBlockId4MemReusedRegst(Plan* plan);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_
