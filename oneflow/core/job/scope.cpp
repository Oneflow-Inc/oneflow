#include "oneflow/core/job/scope.h"

namespace oneflow {

Scope::Scope(const ScopeProto& scope_proto) : scope_proto_(scope_proto) {
  CHECK_OK(Init()) << scope_proto_.DebugString();
}

Maybe<void> Scope::Init() {
  CHECK_OR_RETURN(scope_proto_.has_global_function_conf());
  job_desc_.reset(new JobDesc(scope_proto_.global_function_conf()));
  CHECK_OR_RETURN(scope_proto_.has_parallel_conf());
  parallel_desc_.reset(new ParallelDesc(scope_proto_.parallel_conf()));
  return Maybe<void>::Ok();
}

Maybe<const JobDesc*> Scope::job_desc() const {
  CHECK_NOTNULL_OR_RETURN(job_desc_.get());
  return job_desc_.get();
}

Maybe<const ParallelDesc*> Scope::parallel_desc() const {
  CHECK_NOTNULL_OR_RETURN(parallel_desc_.get());
  return parallel_desc_.get();
}

}  // namespace oneflow
