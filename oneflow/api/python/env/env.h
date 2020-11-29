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
#include "oneflow/api/python/env/env_helper.h"

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>> CurrentResource() {
  return oneflow::CurrentResource().GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>> EnvResource() {
  return oneflow::EnvResource().GetDataAndErrorProto(std::string(""));
}

std::shared_ptr<oneflow::cfg::ErrorProto> EnableEagerEnvironment(bool enable_eager_execution) {
  return oneflow::EnableEagerEnvironment(enable_eager_execution).GetDataAndErrorProto();
}

std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> IsEnvInited() {
  return oneflow::IsEnvInited().GetDataAndErrorProto(false);
}

std::shared_ptr<oneflow::cfg::ErrorProto> InitEnv(const std::string& env_proto_str) {
  return oneflow::InitEnv(env_proto_str).GetDataAndErrorProto();
}

std::shared_ptr<oneflow::cfg::ErrorProto> DestroyEnv() {
  return oneflow::DestroyEnv().GetDataAndErrorProto();
}

std::pair<long long, std::shared_ptr<oneflow::cfg::ErrorProto>> CurrentMachineId() {
  return oneflow::CurrentMachineId().GetDataAndErrorProto(0LL);
}
