#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_set_compile_ctx.h"
#include "oneflow/core/job/runtime_buffer_managers_scope.h"

namespace oneflow {

Session::Session() {
  Global<JobName2JobId>::New();
  Global<CriticalSectionDesc>::New();
  Global<InterUserJobInfo>::New();
  Global<JobBuildAndInferCtxMgr>::New();
  Global<JobSetCompileCtx>::New();
  Global<RuntimeBufferManagersScope>::New();
}

Session::~Session() {
  Global<RuntimeBufferManagersScope>::Delete();
  Global<JobSetCompileCtx>::Delete();
  Global<JobBuildAndInferCtxMgr>::Delete();
  Global<InterUserJobInfo>::Delete();
  Global<CriticalSectionDesc>::Delete();
  Global<JobName2JobId>::Delete();
  Global<CtrlClient>::Get()->Clear();
}

}  // namespace oneflow
