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
#ifndef ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_
#define ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class MachineCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MachineCtx);
  MachineCtx() = delete;
  ~MachineCtx() = default;

  int64_t this_machine_id() const { return this_machine_id_; }
  bool IsThisMachineMaster() const { return this_machine_id_ == 0; }
  std::string GetThisCtrlAddr() const { return GetCtrlAddr(this_machine_id_); }
  std::string GetMasterCtrlAddr() const { return GetCtrlAddr(0); }
  std::string GetCtrlAddr(int64_t machine_id) const;

 private:
  friend class Global<MachineCtx>;
  explicit MachineCtx(int64_t this_mchn_id);

  int64_t this_machine_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_
