#ifndef ONEFLOW_API_JAVA_ENV_ENV_API_H_
#define ONEFLOW_API_JAVA_ENV_ENV_API_H_

#include <string>
#include "oneflow/api/python/env/env.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/scope.h"


inline void InitEnv(int ctrl_port) {
  oneflow::EnvProto env_proto;
  auto* machine = env_proto.add_machine();
  machine->set_id(0);
  machine->set_addr("127.0.0.1");
  env_proto.set_ctrl_port(ctrl_port);

  std::cout << env_proto.DebugString() << std::endl;
  return oneflow::InitEnv(env_proto.DebugString(), false).GetOrThrow();
}

inline long long CurrentMachineId() {
  return oneflow::CurrentMachineId().GetOrThrow(); 
}

inline void InitScopeStack() {
  std::shared_ptr<oneflow::cfg::JobConfigProto> job_conf = std::make_shared<oneflow::cfg::JobConfigProto>();
  job_conf->mutable_predict_conf();
  job_conf->set_job_name("");

  std::shared_ptr<oneflow::Scope> scope;
  auto BuildInitialScope = [&scope, &job_conf](oneflow::InstructionsBuilder* builder) mutable -> oneflow::Maybe<void> {
    // default configuration
    int session_id = oneflow::GetDefaultSessionId().GetOrThrow();
    const std::vector<std::string> machine_device_ids({"0:0"});
    std::shared_ptr<oneflow::Scope> initialScope = builder->BuildInitialScope(session_id, job_conf, "cpu", machine_device_ids, nullptr, false).GetPtrOrThrow();
    scope = initialScope;
    return oneflow::Maybe<void>::Ok();
  };
  oneflow::LogicalRun(BuildInitialScope);
  oneflow::InitThreadLocalScopeStack(scope);  // fixme: bug? LogicalRun is asynchronous
}

#endif  // ONEFLOW_API_JAVA_ENV_ENV_API_H_
