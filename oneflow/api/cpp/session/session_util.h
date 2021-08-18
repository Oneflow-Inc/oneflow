#ifndef ONEFLOW_API_CPP_ENV_SESSION_UTIL_H_
#define ONEFLOW_API_CPP_ENV_SESSION_UTIL_H_

#include <string>
#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/framework/session_util.h"

namespace oneflow {

ConfigProto GetDefaultConfigProto() {
    ConfigProto config_proto;
    config_proto.mut_resource()->set_machine_num(0);
#ifdef WITH_CUDA
    config_proto.mut_resource()->set_gpu_device_num(1);
#else
    config_proto.mut_resource()->set_cpu_device_num(1);
    config_proto.mut_resource()->set_gpu_device_num(0);
#endif  // WITH_CUDA
    int session_id = GetDefaultSessionId().GetOrThrow();
    config_proto.set_session_id(session_id);
    return config_proto;
}

void TryCompleteConfigProto(ConfigProto& config_proto) {
    if (config_proto.resource().machine_num() == 0) {
        config_proto.mut_resource()->set_machine_num(GetNodeSize());
    }
}

inline Maybe<void> InitLazyGlobalSession(const ConfigProto& config_proto) {
  CHECK_NOTNULL_OR_RETURN(Global<EnvDesc>::Get()) << "env not found";
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());

  ClusterInstruction::MasterSendSessionStart();

  FixCpuDeviceNum(&config_proto);
  Global<CtrlClient>::Get()->PushKV("config_proto", config_proto);

  CHECK_ISNULL_OR_RETURN(Global<SessionGlobalObjectsScope>::Get());
  Global<SessionGlobalObjectsScope>::SetAllocated(new SessionGlobalObjectsScope());
  JUST(Global<SessionGlobalObjectsScope>::Get()->Init(config_proto));
  LOG(INFO) << "NewGlobal " << typeid(SessionGlobalObjectsScope).name();
  return Maybe<void>::Ok();
}

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_ENV_SESSION_UTIL_H_