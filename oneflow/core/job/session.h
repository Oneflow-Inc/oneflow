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
#ifndef ONEFLOW_CORE_JOB_SESSION_H_
#define ONEFLOW_CORE_JOB_SESSION_H_

#include <memory>
#include <string>

namespace oneflow {

int64_t NewSessionId();

class ConfigProto;
class ConfigProtoContext {
 public:
  ConfigProtoContext(const ConfigProto& config_proto);
  ~ConfigProtoContext();

  int64_t session_id() const { return session_id_; }

 private:
  int64_t session_id_;
};

class LogicalConfigProtoContext {
 public:
  LogicalConfigProtoContext(const std::string& config_proto_str);
  ~LogicalConfigProtoContext();

  std::unique_ptr<ConfigProtoContext> config_proto_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SESSION_H_
