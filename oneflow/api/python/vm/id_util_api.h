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

#include <atomic>
#include "oneflow/api/python/vm/id_util.h"

inline long long NewLogicalObjectId() { return oneflow::NewLogicalObjectId().GetOrThrow(); }

inline long long NewLogicalSymbolId() { return oneflow::NewLogicalSymbolId().GetOrThrow(); }

inline long long NewPhysicalObjectId() { return oneflow::NewPhysicalObjectId().GetOrThrow(); }

inline long long NewPhysicalSymbolId() { return oneflow::NewPhysicalSymbolId().GetOrThrow(); }

inline uint64_t NewTokenId() {
  static std::atomic<uint64_t> token_id(0);
  token_id++;
  return token_id;
}

#endif  // ONEFLOW_API_PYTHON_VM_ID_UTIL_API_H_
