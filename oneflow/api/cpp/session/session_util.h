#ifndef ONEFLOW_API_CPP_ENV_SESSION_UTIL_H_
#define ONEFLOW_API_CPP_ENV_SESSION_UTIL_H_

#include <string>
#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/framework/session_util.h"

oneflow::ConfigProto GetDefaultConfigProto() {
    oneflow::ConfigProto config_proto;
    config_proto.mut_resource()->set_machine_num(0);
#ifdef WITH_CUDA
    config_proto.mut_resource()->set_gpu_device_num(1);
#else
    config_proto.mut_resource()->set_cpu_device_num(1);
    config_proto.mut_resource()->set_gpu_device_num(0);
#endif  // WITH_CUDA
    int session_id = oneflow::GetDefaultSessionId().GetOrThrow();
    config_proto.set_session_id(session_id);
    return config_proto;
}

void TryCompleteConfigProto(oneflow::ConfigProto& config_proto) {
    if (config_proto.resource().machine_num() == 0) {
        config_proto.mut_resource()->set_machine_num(GetNodeSize());
    }
}

#endif  // ONEFLOW_API_CPP_ENV_SESSION_UTIL_H_