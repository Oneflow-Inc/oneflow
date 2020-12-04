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
#include "oneflow/core/actor/normal_forward_compute_actor.h"

namespace oneflow {

void NormalForwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  cur_piece_id_ = -1;
  const_buf_regst_desc_id_ = Name2SoleRegstDescId("const_buf");
  const_buf_regst_ = nullptr;
  if (const_buf_regst_desc_id_ != -1) {
    const_buf_regst_ = GetSoleProducedRegst4RegstDescId(const_buf_regst_desc_id_);
    AsyncInitModelAndConstBuf();
  }
  OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerNormal);
}

void NormalForwardCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  cur_piece_id_ = GetPieceId4NaiveOrInplaceCurReadableDataRegst();
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == const_buf_regst_desc_id_) {
      return const_buf_regst_;
    } else {
      return nullptr;
    }
  });
}

void NormalForwardCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
}

void NormalForwardCompActor::VirtualAsyncSendInplaceProducedRegstMsgToConsumer() {
  HandleProducedInplaceDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
}

void NormalForwardCompActor::AsyncInitModelAndConstBuf() {
  for (const ExecKernel& exec_kernel : exec_kernel_vec()) {
    KernelCtx kernel_ctx = GenDefaultKernelCtx();
    exec_kernel.kernel->InitModelAndConstBuf(kernel_ctx, [&](const std::string& bn_in_op) {
      const LogicalBlobId& lbi = exec_kernel.kernel->BnInOp2Lbi(bn_in_op);
      Blob* blob = nullptr;
      if (const_buf_regst_) { blob = const_buf_regst_->GetBlobByLbi(lbi); }
      return blob;
    });
  }
}

REGISTER_ACTOR(TaskType::kNormalForward, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kPrint, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kForeignInput, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kForeignOutput, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kDistributeConcat, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kDistributeSplit, NormalForwardCompActor);

}  // namespace oneflow
