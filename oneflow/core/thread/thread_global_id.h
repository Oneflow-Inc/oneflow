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
#ifndef ONEFLOW_CORE_THREAD_GLOBAL_UNIQUE_ID_H_
#define ONEFLOW_CORE_THREAD_GLOBAL_UNIQUE_ID_H_

#include <string>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"

namespace oneflow {

const static int kThreadGlobalIdDefaultWorker = 0;
const static int kThreadGlobalIdMain = 7;

int64_t GetThisThreadGlobalId();

class ThreadGlobalIdGuard final {
 public:
  explicit ThreadGlobalIdGuard(int64_t thread_global_id);
  ~ThreadGlobalIdGuard();

 private:
  int64_t old_thread_global_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_GLOBAL_UNIQUE_ID_H_
