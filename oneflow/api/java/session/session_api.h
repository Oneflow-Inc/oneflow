#ifndef ONEFLOW_API_JAVA_SESSION_SESSION_API_H_
#define ONEFLOW_API_JAVA_SESSION_SESSION_API_H_

#include <string>
#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/framework/session_util.h"

inline void OpenDefaultSession() {
    int64_t session_id = oneflow::NewSessionId();
    oneflow::RegsiterSession(session_id);
}

inline void InitSession(const std::string& config_proto) {
  oneflow::InitLazyGlobalSession(config_proto);
}

#endif  // ONEFLOW_API_JAVA_SESSION_SESSION_API_H_
