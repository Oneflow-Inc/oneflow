#ifndef ONEFLOW_API_JAVA_SESSION_SESSION_API_H_
#define ONEFLOW_API_JAVA_SESSION_SESSION_API_H_

#include <string>
#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/framework/session_util.h"

inline void OpenDefaultSession() {
    int64_t session_id = oneflow::NewSessionId();
    oneflow::RegsiterSession(session_id);
}

inline void InitSession(const std::string& device_tag) {
  oneflow::ConfigProto config_proto;
  config_proto.set_session_id(0);
  auto* resource = config_proto.mutable_resource();
  resource->set_machine_num(1);
  resource->set_enable_legacy_model_io(true);
  if (device_tag == "gpu") {
    resource->set_gpu_device_num(1);
  }
  else {
    resource->set_cpu_device_num(1);
  }

  std::cout << config_proto.DebugString() << std::endl;
  oneflow::InitLazyGlobalSession(oneflow::PbMessage2TxtString(config_proto));
}

#endif  // ONEFLOW_API_JAVA_SESSION_SESSION_API_H_
