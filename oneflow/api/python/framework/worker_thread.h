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
#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_WORKER_THREAD_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_WORKER_THREAD_H_

namespace oneflow {

class WorkerThread final {
 public:
  explicit WorkerThread(int64_t worker_thread_id) : worker_thread_id_(worker_thread_id) {}

  int64_t worker_thread_id() const { return worker_thread_id_; }

  static Maybe<WorkerThread> New(int64_t worker_thread_id) {
    CHECK_GE_OR_RETURN(worker_thread_id, 0)
        << "worker_thread_id should be >= 0. (current: " << worker_thread_id << ").";
    CHECK_LT_OR_RETURN(worker_thread_id, WorkerThreadMaxSize())
        << "worker_thread_id should be < 4. (current: " << worker_thread_id << ").";
    return std::make_shared<WorkerThread>(worker_thread_id);
  }

  static int WorkerThreadMaxSize() { return 4; }

 private:
  int64_t worker_thread_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_WORKER_THREAD_H_
