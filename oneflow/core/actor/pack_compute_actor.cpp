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
#include "oneflow/core/actor/pack_compute_actor.h"
#include "oneflow/core/kernel/user_kernel.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

void PackCompActor::VirtualCompActorInit(const TaskProto& proto) {
  int64_t out_diff_regst_desc_id = Name2SoleRegstDescId("out_diff");
  handle_unpack_bw_ = out_diff_regst_desc_id != -1;
  if (handle_unpack_bw_) {
    const Shape& out_diff_time_shape = Global<RegstMgr>::Get()
                                           ->RegstDesc4RegstDescId(out_diff_regst_desc_id)
                                           .data_regst_time_shape();
    total_pack_num_ = out_diff_time_shape.At(out_diff_time_shape.NumAxes() - 1);
  } else {
    const Shape& in_time_shape = Global<RegstMgr>::Get()
                                     ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                     .data_regst_time_shape();
    total_pack_num_ = in_time_shape.At(in_time_shape.NumAxes() - 1);
  }
  act_num_cnt_ = 0;
  cur_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&PackCompActor::HandlerNormal);
}

void PackCompActor::Act() {
  KernelCtx ctx = GenDefaultKernelCtx();
  CHECK_GE(exec_kernel_vec().size(), 1);
  auto user_kernel = dynamic_cast<const UserKernel*>(exec_kernel_vec().at(0).kernel.get());
  CHECK_NOTNULL(user_kernel);
  auto state = dynamic_cast<OpKernelStateWrapper<std::pair<size_t, size_t>>*>(
      user_kernel->GetOpKernelState().get());
  CHECK_NOTNULL(state);
  state->Mutable()->first = act_num_cnt_;
  state->Mutable()->second = total_pack_num_;
  AsyncLaunchKernel(ctx);
  act_num_cnt_ += 1;
}

void PackCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (act_num_cnt_ == total_pack_num_) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      regst->set_piece_id(cur_piece_id_);
      return true;
    });
    cur_piece_id_ += 1;
  }
}

void PackCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (handle_unpack_bw_ == false) {
    HandleConsumedNaiveDataRegstToProducer([](Regst*) { return true; });
  } else {
    int64_t in_regst_desc_id = Name2SoleRegstDescId("in");
    HandleConsumedNaiveDataRegstToProducer(
        [in_regst_desc_id](Regst* regst) { return regst->regst_desc_id() != in_regst_desc_id; });
    if (act_num_cnt_ == total_pack_num_) {
      AsyncSendRegstMsgToProducer(GetNaiveCurReadable(in_regst_desc_id));
    }
  }
  if (act_num_cnt_ == total_pack_num_) { act_num_cnt_ = 0; }
}

REGISTER_ACTOR(TaskType::kPack, PackCompActor);

}  // namespace oneflow
