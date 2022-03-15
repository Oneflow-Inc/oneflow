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

#ifndef ONEFLOW_CORE_FRAMEWORK_GLOBAL_PARAM_GRAD_SYNC_MODE_
#define ONEFLOW_CORE_FRAMEWORK_GLOBAL_PARAM_GRAD_SYNC_MODE_

namespace oneflow {

struct GlobalGradSyncMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

class GlobalParamGradSyncMode {
 public:
  GlobalParamGradSyncMode(bool enabled) : prev_mode_(GlobalGradSyncMode::is_enabled()) {
    GlobalGradSyncMode::set_enabled(enabled);
  }
  ~GlobalParamGradSyncMode() { GlobalGradSyncMode::set_enabled(prev_mode_); }
  bool prev_mode() const { return prev_mode_; }

 private:
  bool prev_mode_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_GLOBAL_PARAM_GRAD_SYNC_MODE_
