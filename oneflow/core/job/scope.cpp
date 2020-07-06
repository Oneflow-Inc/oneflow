#include "oneflow/core/job/scope.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

Scope::Scope(const ScopeProto& scope_proto) : scope_proto_(scope_proto) {
  CHECK_OK(Init()) << scope_proto_.DebugString();
}

Maybe<void> Scope::Init() {
  {
    const auto& storage = *Global<vm::SymbolStorage<JobDesc>>::Get();
    job_desc_ = storage.GetPtr(scope_proto_.job_desc_symbol_id());
  }
  {
    const auto& storage = *Global<vm::SymbolStorage<ParallelDesc>>::Get();
    device_parallel_desc_ = storage.GetPtr(scope_proto_.device_parallel_desc_symbol_id());
    host_parallel_desc_ = storage.GetPtr(scope_proto_.host_parallel_desc_symbol_id());
  }
  return Maybe<void>::Ok();
}

Maybe<const JobDesc*> Scope::job_desc() const {
  CHECK_NOTNULL_OR_RETURN(job_desc_.get());
  return job_desc_.get();
}

Maybe<const ParallelDesc*> Scope::GetParallelDesc(const OperatorConf& op_conf) const {
  if (IsClassRegistered<OnlyCpuSupportPredicator>(op_conf.op_type_case())) {
    return host_parallel_desc_.get();
  } else {
    return device_parallel_desc_.get();
  }
}

}  // namespace oneflow
