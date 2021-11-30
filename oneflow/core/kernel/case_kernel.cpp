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
#include "oneflow/core/kernel/case_kernel.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

template<typename T>
void CaseKernel<T>::VirtualKernelInit(KernelContext* ctx) {
  ctx->set_state(std::make_shared<CaseStatus>());
}

template<typename T>
void CaseKernel<T>::ForwardDataContent(KernelContext* ctx) const {
  CaseStatus* const case_status = CHECK_NOTNULL(dynamic_cast<CaseStatus*>(ctx->state().get()));
  if (case_status->cmd == kCaseCmdHandleInput) {
    int64_t cur_selected_id = static_cast<int64_t>(ctx->BnInOp2Blob("in")->dptr<T>()[0]);
    case_status->select_id2request_cnt[cur_selected_id] += 1;
  } else if (case_status->cmd == kCaseCmdHandleOutput) {
    int64_t cur_selected_id = case_status->cur_selected_id;
    CHECK_GT(case_status->select_id2request_cnt[cur_selected_id], 0);
    case_status->select_id2request_cnt[cur_selected_id] -= 1;
    if (case_status->select_id2request_cnt[cur_selected_id] == 0) {
      case_status->select_id2request_cnt.erase(cur_selected_id);
    }
    *(ctx->BnInOp2Blob(GenRepeatedBn("out", cur_selected_id))->mut_dptr<T>()) = cur_selected_id;
  } else {
    UNIMPLEMENTED();
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCaseConf, CaseKernel, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
