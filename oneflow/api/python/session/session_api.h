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

inline std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> IsSessionInited() {
  return oneflow::IsSessionInited().GetDataAndErrorProto(false);
}

inline std::shared_ptr<oneflow::cfg::ErrorProto> InitLazyGlobalSession(
    const std::string& config_proto_str) {
  return oneflow::InitLazyGlobalSession(config_proto_str).GetDataAndErrorProto();
}

inline std::shared_ptr<oneflow::cfg::ErrorProto> DestroyLazyGlobalSession() {
  return oneflow::DestroyLazyGlobalSession().GetDataAndErrorProto();
}

inline std::shared_ptr<oneflow::cfg::ErrorProto> StartLazyGlobalSession() {
  return oneflow::StartLazyGlobalSession().GetDataAndErrorProto();
}

inline std::shared_ptr<oneflow::cfg::ErrorProto> StopLazyGlobalSession() {
  return oneflow::StopLazyGlobalSession().GetDataAndErrorProto();
}

#endif  // ONEFLOW_API_PYTHON_SESSION_SESSION_API_H_
