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
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/framework/thread.h"
#include "oneflow/core/common/env_var/vm.h"

namespace py = pybind11;

namespace oneflow {

namespace {

class UsingThreadUidSet final {
 public:
  UsingThreadUidSet()
      : using_thread_uids_({Stream::kDefaultStreamThreadUid}),
        thread_limits_(using_thread_uids_.size()
                       + ThreadLocalEnvInteger<ONEFLOW_VM_WORKER_THREAD_LIMIT>()) {}
  ~UsingThreadUidSet() = default;

  Maybe<int64_t> Get() {
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK_LT_OR_RETURN(using_thread_uids_.size(), thread_limits_)
        << "can not create more worker threads. please check your code or increase environment "
           "variable ONEFLOW_VM_WORKER_THREAD_LIMIT(default value:"
        << ThreadLocalEnvInteger<ONEFLOW_VM_WORKER_THREAD_LIMIT>() << ")";
    for (int i = 0; i < using_thread_uids_.size() + 1; ++i) {
      if (using_thread_uids_.count(i) == 0) {
        using_thread_uids_.insert(i);
        return i;
      }
    }
    UNIMPLEMENTED_THEN_RETURN();
  }

  Maybe<void> Put(int64_t thread_uid) {
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK_NE_OR_RETURN(thread_uid, Stream::kDefaultStreamThreadUid)
        << "default thread_uid should not be erased. value: " << thread_uid;
    CHECK_OR_RETURN(using_thread_uids_.erase(thread_uid) > 0)
        << "no thread_uid found. (current: " << thread_uid << ").";
    return Maybe<void>::Ok();
  }

 private:
  std::set<int64_t> using_thread_uids_;
  size_t thread_limits_;
  std::mutex mutex_;
};

UsingThreadUidSet* MutUsingThreadUidSet() {
  static UsingThreadUidSet thread_uid_set;
  return &thread_uid_set;
}

}  // namespace

/*static*/ Maybe<AsyncThread> AsyncThread::New() {
  return std::shared_ptr<AsyncThread>(new AsyncThread(JUST(MutUsingThreadUidSet()->Get())));
}

AsyncThread::~AsyncThread() { MutUsingThreadUidSet()->Put(thread_uid_).GetOrThrow(); }

}  // namespace oneflow

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;
  py::class_<AsyncThread, std::shared_ptr<AsyncThread>>(m, "AsyncThread").def(py::init([]() {
    return AsyncThread::New().GetPtrOrThrow();
  }));
}
