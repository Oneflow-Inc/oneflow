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
#include "oneflow/core/actor/unpack_compute_actor.h"
#include "oneflow/core/kernel/user_kernel.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

void UnpackCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& out_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                    .data_regst_time_shape();
  total_unpack_num_ = out_time_shape.At(out_time_shape.NumAxes() - 1);
  act_num_cnt_ = 0;
  cur_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&UnpackCompActor::HandlerNormal);
}

void UnpackCompActor::Act() {
  KernelCtx ctx = GenDefaultKernelCtx();
  CHECK_GE(exec_kernel_vec().size(), 1);
  auto user_kernel = dynamic_cast<const UserKernel*>(exec_kernel_vec().at(0).kernel.get());
  CHECK_NOTNULL(user_kernel);
  auto state = dynamic_cast<OpKernelStateWrapper<std::pair<size_t, size_t>>*>(
      user_kernel->GetOpKernelState().get());
  CHECK_NOTNULL(state);
  state->Mutable()->first = act_num_cnt_;
  state->Mutable()->second = total_unpack_num_;
  AsyncLaunchKernel(ctx);
  act_num_cnt_ += 1;
}

void UnpackCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
  cur_piece_id_ += 1;
}

void UnpackCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (act_num_cnt_ == total_unpack_num_) {
    HandleConsumedNaiveDataRegstToProducer([](Regst*) { return true; });
    act_num_cnt_ = 0;
  }
}

bool UnpackCompActor::ConsumedCtrlRegstValid(int64_t regst_desc_id) const {
  return act_num_cnt_ == 0;
}

REGISTER_ACTOR(TaskType::kUnpack, UnpackCompActor);

}  // namespace oneflow
