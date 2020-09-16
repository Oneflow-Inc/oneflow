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
#include "oneflow/core/thread/glog_failure_function.h"

namespace oneflow {

namespace {

void PyFailureFunction() {
  const int ret = Global<GlogFailureFunction>::Get()->RunCallback("panic in C++");
  const bool is_main_thread = Global<GlogFailureFunction>::Get()->IsMainThread();
  if (ret == kMainThreadPanic && is_main_thread) {
    throw MainThreadPanic();
  } else if (ret == kWorkerThreadPanic && is_main_thread == false) {
    throw WorkerThreadPanic();
  } else {
    LOG(ERROR) << "failure callback return code: " << ret << ", is_main_thread: " << is_main_thread
               << ", aborting";
    abort();
  }
}

}  // namespace

void GlogFailureFunction::SetCallback(const py_failure_callback& f) {
  failure_function_ = f;
  is_function_set_ = true;
  UpdateThreadLocal();
}

void GlogFailureFunction::UpdateThreadLocal() {
  if (is_function_set_) { google::InstallFailureFunction(&PyFailureFunction); }
}

}  // namespace oneflow
