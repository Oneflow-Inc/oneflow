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
#ifndef ONEFLOW_CORE_CONTROL_CTRL_GLOBAL_PROCESS_RANK_INFO_H_
#define ONEFLOW_CORE_CONTROL_CTRL_GLOBAL_PROCESS_RANK_INFO_H_

#include <string>

namespace oneflow {

struct GlobalProcessCtx {
  static int64_t ThisProcessId();
  static bool IsThisProcessMaster();
  static size_t TotalProcessNum();
  static std::string GetCtrlAddr(int64_t process_id);
  static std::string GetThisCtrlAddr();
  static std::string GetMasterCtrlAddr();
  static std::string PersistentPathPrefix();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_GLOBAL_PROCESS_RANK_INFO_H_
