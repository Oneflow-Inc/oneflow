/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
    resource->set_cpu_device_num(0);
    resource->set_gpu_device_num(1);
  } else {
    resource->set_cpu_device_num(1);
    resource->set_gpu_device_num(0);
  }
  oneflow::InitLazyGlobalSession(oneflow::PbMessage2TxtString(config_proto));
}

#endif  // ONEFLOW_API_JAVA_SESSION_SESSION_API_H_
