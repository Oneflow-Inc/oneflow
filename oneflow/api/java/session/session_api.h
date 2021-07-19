#ifndef ONEFLOW_API_JAVA_SESSION_SESSION_API_H_
#define ONEFLOW_API_JAVA_SESSION_SESSION_API_H_

#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/framework/session_util.h"

inline void OpenDefaultSession() {
    int64_t session_id = oneflow::NewSessionId();
    oneflow::RegsiterSession(session_id);
}

inline void InitSession() {
  // default configuration
  // reference: oneflow/python/framework/session_util.py
  std::shared_ptr<oneflow::ConfigProto> config_proto = std::make_shared<oneflow::ConfigProto>();
  config_proto->mutable_resource()->set_machine_num(oneflow::GetNodeSize().GetOrThrow());
  config_proto->mutable_resource()->set_gpu_device_num(1);
  config_proto->mutable_resource()->set_enable_legacy_model_io(true);
  config_proto->set_session_id(oneflow::GetDefaultSessionId().GetOrThrow());

  oneflow::InitLazyGlobalSession(config_proto->DebugString());
}

#endif  // ONEFLOW_API_JAVA_SESSION_SESSION_API_H_
