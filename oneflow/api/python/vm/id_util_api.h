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
#ifndef ONEFLOW_API_PYTHON_VM_ID_UTIL_API_H_
#define ONEFLOW_API_PYTHON_VM_ID_UTIL_API_H_

#include "oneflow/api/python/vm/id_util.h"

inline std::pair<long long, std::shared_ptr<oneflow::cfg::ErrorProto>> NewLogicalObjectId() {
  return oneflow::NewLogicalObjectId().GetDataAndErrorProto(0LL);
}

inline std::pair<long long, std::shared_ptr<oneflow::cfg::ErrorProto>> NewLogicalSymbolId() {
  return oneflow::NewLogicalSymbolId().GetDataAndErrorProto(0LL);
}

inline std::pair<long long, std::shared_ptr<oneflow::cfg::ErrorProto>> NewPhysicalObjectId() {
  return oneflow::NewPhysicalObjectId().GetDataAndErrorProto(0LL);
}

inline std::pair<long long, std::shared_ptr<oneflow::cfg::ErrorProto>> NewPhysicalSymbolId() {
  return oneflow::NewPhysicalSymbolId().GetDataAndErrorProto(0LL);
}

#endif  // ONEFLOW_API_PYTHON_VM_ID_UTIL_API_H_
