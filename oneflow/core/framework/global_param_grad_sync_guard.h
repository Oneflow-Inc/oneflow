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
#ifndef ONEFLOW_CORE_FRAMEWORK_GLOBAL_PARAM_GRAD_SYNC_GUARD_H_
#define ONEFLOW_CORE_FRAMEWORK_GLOBAL_PARAM_GRAD_SYNC_GUARD_H_

namespace oneflow {

class GlobalParamGardSyncGuard {
 public:
  GlobalParamGardSyncGuard(bool flag) {
    old_flag_ = flag_;
    flag_ = flag;
  }
  ~GlobalParamGardSyncGuard() { flag_ = old_flag_; }

  static bool* MutThreadLocalGlobalParamGradSyncFlag() { return &flag_; }

 private:
  static thread_local bool old_flag_;
  static thread_local bool flag_;
};

thread_local bool GlobalParamGardSyncGuard::old_flag_ = true;
thread_local bool GlobalParamGardSyncGuard::flag_ = true;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_GLOBAL_PARAM_GRAD_SYNC_GUARD_H_
