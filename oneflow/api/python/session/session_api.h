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
#ifndef ONEFLOW_API_PYTHON_SESSION_SESSION_API_H_
#define ONEFLOW_API_PYTHON_SESSION_SESSION_API_H_

#include "oneflow/api/python/session/session.h"

inline bool IsSessionInited() { return oneflow::IsSessionInited().GetOrThrow(); }

inline void InitEagerGlobalSession(const std::string& config_proto_str) {
  return oneflow::InitEagerGlobalSession(config_proto_str).GetOrThrow();
}

inline void InitLazyGlobalSession(const std::string& config_proto_str) {
  return oneflow::InitLazyGlobalSession(config_proto_str).GetOrThrow();
}

inline void DestroyLazyGlobalSession() { return oneflow::DestroyLazyGlobalSession().GetOrThrow(); }

inline void StartLazyGlobalSession() { return oneflow::StartLazyGlobalSession().GetOrThrow(); }

inline void StopLazyGlobalSession() { return oneflow::StopLazyGlobalSession().GetOrThrow(); }

#endif  // ONEFLOW_API_PYTHON_SESSION_SESSION_API_H_
