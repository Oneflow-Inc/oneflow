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

#include "oneflow/core/kernel/wait_and_send_ids_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

template<typename T>
void WaitAndSendIdsKernel<T>::VirtualKernelInit(KernelContext* ctx) {
  ctx->set_state(new WaitAndSendIdsStatus);
}

template<typename T>
void WaitAndSendIdsKernel<T>::DestroyState(void* state) const {
  delete static_cast<WaitAndSendIdsStatus*>(state);
}

template<typename T>
void WaitAndSendIdsKernel<T>::ForwardDataContent(KernelContext* ctx) const {
  CHECK(ctx->state());
  auto* status = static_cast<WaitAndSendIdsStatus*>(ctx->state());
  const auto& conf = this->op_conf().wait_and_send_ids_conf();
  if (status->out_idx_ >= status->out_num_) {
    if (CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get())) {
      CHECK(this->op_conf().wait_and_send_ids_conf().has_job_name());
      const auto& job_name = this->op_conf().wait_and_send_ids_conf().job_name();
      auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
      auto* buffer = buffer_mgr->Get(GetSourceTickBufferName(job_name));
      status->in_id_ = 0;
      {
        std::shared_ptr<JobInstance> job_instance;
        status->buffer_status_ = buffer->Receive(&job_instance);
      }
      if (status->buffer_status_ == kBufferStatusErrorClosed) { return; }
      status->out_idx_ = 0;
      status->out_num_ = 1;
    } else {
      auto* buffer_mgr = Global<BufferMgr<int64_t>>::Get();
      status->buffer_status_ = buffer_mgr->Get(conf.wait_buffer_name())->Receive(&status->in_id_);
      if (status->buffer_status_ == kBufferStatusErrorClosed) { return; }
      status->out_idx_ = 0;
      status->out_num_ = conf.id_list(status->in_id_).value_size();
    }
  }

  if (CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get())) {
    *ctx->BnInOp2Blob("out")->mut_dptr<T>() = 0;
  } else {
    *ctx->BnInOp2Blob("out")->mut_dptr<T>() = conf.id_list(status->in_id_).value(status->out_idx_);
  }
  ++status->out_idx_;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kWaitAndSendIdsConf, WaitAndSendIdsKernel,
                               INT_DATA_TYPE_SEQ);

}  // namespace oneflow
