#ifndef ONEFLOW_API_CPP_FRAMEWORK_FRAMEWORK_UTIL_H_
#define ONEFLOW_API_CPP_FRAMEWORK_FRAMEWORK_UTIL_H_

#include <string>
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"

namespace oneflow {

inline InterUserJobInfo* GetInterUserJobInfo() {
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());
  CHECK_NOTNULL_OR_RETURN(Global<Oneflow>::Get());
  CHECK_NOTNULL_OR_RETURN(Global<InterUserJobInfo>::Get());
  return Global<InterUserJobInfo>::Get();
}

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_FRAMEWORK_FRAMEWORK_UTIL_H_