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
#include <atomic>
#include "oneflow/core/job/session.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

int64_t NewSessionId() {
  static std::atomic<int64_t> counter(0);
  return counter++;
}

ConfigProtoContext::ConfigProtoContext(const ConfigProto& config_proto)
    : session_id_(config_proto.session_id()) {
  Global<const IOConf>::SessionNew(session_id_, config_proto.io_conf());
}

ConfigProtoContext::~ConfigProtoContext() { Global<const IOConf>::SessionDelete(session_id_); }

LogicalConfigProtoContext::LogicalConfigProtoContext(const std::string& config_proto_str) {
  ConfigProto config_proto;
  CHECK(TxtString2PbMessage(config_proto_str, &config_proto));
  // TODO(hanbinbin): init for worker machines
  config_proto_ctx_.reset(new ConfigProtoContext(config_proto));
}

LogicalConfigProtoContext::~LogicalConfigProtoContext() {
  config_proto_ctx_.reset();
  // TODO(hanbinbin): destory ConfigProtoContext of worker machines
}

}  // namespace oneflow
