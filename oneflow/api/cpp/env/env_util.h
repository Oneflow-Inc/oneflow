#ifndef ONEFLOW_API_CPP_ENV_ENV_UTIL_H_
#define ONEFLOW_API_CPP_ENV_ENV_UTIL_H_

#include <string>
#include <memory>
#include "oneflow/api/python/env/env.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {

std::shared_ptr<Scope> MakeInitialScope(std::shared_ptr<cfg::JobConfigProto> job_conf, 
                                        std::string device_tag, 
                                        std::vector<std::string> machine_device_ids, 
                                        std::shared_ptr<Shape> hierarchy, 
                                        bool is_mirrored) {
  std::shared_ptr<Scope> scope;
  auto BuildInitialScope = [&](InstructionsBuilder* builder) mutable -> Maybe<void> {
    // default configuration
    int session_id = GetDefaultSessionId().GetOrThrow();
    std::shared_ptr<Scope> initialScope = 
        builder->BuildInitialScope(session_id, job_conf, device_tag, 
                machine_device_ids, hierarchy, is_mirrored).GetPtrOrThrow();
    scope = initialScope;
    return Maybe<void>::Ok();
  };
  LogicalRun(BuildInitialScope).GetOrThrow();
  return scope;
}
  
inline Maybe<void> InitScopeStack() {
  std::shared_ptr<cfg::JobConfigProto> job_conf = std::make_shared<cfg::JobConfigProto>();
  job_conf->mutable_predict_conf();
  job_conf->set_job_name("");

  const std::vector<std::string> machine_device_ids({"0:0"});
  std::shared_ptr<Scope> scope = MakeInitialScope(job_conf, "cpu", machine_device_ids, nullptr, false);
  return InitThreadLocalScopeStack(scope);  // fixme: bug? LogicalRun is asynchronous
}

inline Maybe<void> InitEnv(const EnvProto& env_proto, bool is_multi_client) {
  CHECK_ISNULL_OR_RETURN(Global<EnvGlobalObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvGlobalObjectsScope>::SetAllocated(new EnvGlobalObjectsScope());
  JUST(Global<EnvGlobalObjectsScope>::Get()->Init(env_proto));
  if (!GlobalProcessCtx::IsThisProcessMaster() && !is_multi_client) { JUST(Cluster::WorkerLoop()); }
  return Maybe<void>::Ok();
}

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_ENV_ENV_UTIL_H_