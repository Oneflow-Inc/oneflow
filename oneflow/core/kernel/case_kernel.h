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
#ifndef ONEFLOW_CORE_KERNEL_CASE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CASE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

enum CaseCmd {
  kCaseCmdInvalid = 0,
  kCaseCmdHandleInput = 1,
  kCaseCmdHandleOutput = 2,
};

struct CaseStatus final : public KernelState {
  CaseStatus() : cmd(kCaseCmdInvalid), cur_selected_id(-1) {}
  ~CaseStatus() = default;

  CaseCmd cmd;
  int64_t cur_selected_id;
  HashMap<int64_t, int64_t> select_id2request_cnt;
};

template<typename T>
class CaseKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CaseKernel);
  CaseKernel() = default;
  ~CaseKernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;
  void ForwardDataContent(KernelContext* ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CASE_KERNEL_H_
