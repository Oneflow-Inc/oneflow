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
#ifndef ONEFLOW_CORE_THREAD_CONSISTENT_UNIQUE_ID_H_
#define ONEFLOW_CORE_THREAD_CONSISTENT_UNIQUE_ID_H_

#include <string>
#include "oneflow/core/common/maybe.h"

namespace oneflow {

const static int kThreadConsistentIdMain = 0;
const static int kThreadConsistentIdHook = 1;
const static int kThreadConsistentIdScheduler = 2;

size_t GetThreadConsistentIdCount();

Maybe<void> InitThisThreadUniqueConsistentId(int64_t thread_consistent_id,
                                             const std::string& debug_string);
Maybe<void> InitThisThreadConsistentId(int64_t thread_consistent_id,
                                       const std::string& debug_string);
Maybe<int64_t> GetThisThreadConsistentId();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_CONSISTENT_UNIQUE_ID_H_
