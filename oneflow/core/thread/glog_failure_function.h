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
#ifndef ONEFLOW_CORE_THREAD_GLOG_FAILURE_FUNCTION_H_
#define ONEFLOW_CORE_THREAD_GLOG_FAILURE_FUNCTION_H_

#include "oneflow/core/common/util.h"
namespace oneflow {

struct MainThreadPanic : public std::exception {};
struct WorkerThreadPanic : public std::exception {};

const int kDefaultPanic = -1;
const int kMainThreadPanic = 0;
const int kWorkerThreadPanic = 1;

class GlogFailureFunction final {
 public:
  using py_failure_callback = std::function<int(std::string)>;
  OF_DISALLOW_COPY_AND_MOVE(GlogFailureFunction);
  GlogFailureFunction() {
    is_function_set_ = false;
    failure_function_ = [](std::string) { return kDefaultPanic; };
    main_thread_id_ = std::this_thread::get_id();
  }
  ~GlogFailureFunction() = default;

  void SetCallback(const py_failure_callback&);
  int RunCallback(std::string err_str) { return failure_function_(err_str); }
  void Clear() { is_function_set_ = false; }
  void UpdateThreadLocal();
  bool IsMainThread() { return std::this_thread::get_id() == main_thread_id_; }

 private:
  py_failure_callback failure_function_;
  bool is_function_set_;
  std::thread::id main_thread_id_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_GLOG_FAILURE_FUNCTION_H_
