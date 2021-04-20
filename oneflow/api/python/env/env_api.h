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
#ifndef ONEFLOW_API_PYTHON_ENV_ENV_API_H_
#define ONEFLOW_API_PYTHON_ENV_ENV_API_H_

#include "oneflow/api/python/env/env.h"

inline std::string CurrentResource() { return oneflow::CurrentResource().GetOrThrow(); }

inline std::string EnvResource() { return oneflow::EnvResource().GetOrThrow(); }

inline void EnableEagerEnvironment(bool enable_eager_execution) {
  return oneflow::EnableEagerEnvironment(enable_eager_execution).GetOrThrow();
}

inline bool IsEnvInited() { return oneflow::IsEnvInited().GetOrThrow(); }

inline void InitEnv(const std::string& env_proto_str) {
  return oneflow::InitEnv(env_proto_str).GetOrThrow();
}

inline void DestroyEnv() { return oneflow::DestroyEnv().GetOrThrow(); }

inline long long CurrentMachineId() { return oneflow::CurrentMachineId().GetOrThrow(); }

#endif  // ONEFLOW_API_PYTHON_ENV_ENV_API_H_
