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
#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_THREAD_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_THREAD_H_

#include "oneflow/core/framework/stream.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class AsyncThread final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AsyncThread);
  ~AsyncThread();

  static Maybe<AsyncThread> New();

  int64_t thread_uid() const { return thread_uid_; }

 private:
  AsyncThread(int64_t thread_uid) : thread_uid_(thread_uid) {}

  int64_t thread_uid_;
};

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_THREAD_H_
