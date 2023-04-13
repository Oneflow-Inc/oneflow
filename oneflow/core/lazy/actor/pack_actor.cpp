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
#include "oneflow/core/lazy/actor/actor.h"
#include "oneflow/core/kernel/user_kernel.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"

namespace oneflow {

class PackActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PackActor);
  PackActor() = default;
  ~PackActor() = default;

 private:
  void VirtualActorInit(const TaskProto& proto) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;

  size_t total_pack_num_;
  size_t act_num_cnt_;
};

void PackActor::VirtualActorInit(const TaskProto& proto) {
  const Shape& in_time_shape = Singleton<RegstMgr>::Get()
                                   ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                   .data_regst_time_shape();
  total_pack_num_ = in_time_shape.At(in_time_shape.NumAxes() - 1);
  act_num_cnt_ = 0;
  OF_SET_MSG_HANDLER(&PackActor::HandlerNormal);
}

void PackActor::Act() {
  CHECK_GE(exec_kernel_vec().size(), 1);
  auto user_kernel = dynamic_cast<const UserKernel*>(exec_kernel_vec().at(0).kernel.get());
  CHECK_NOTNULL(user_kernel);
  auto state = dynamic_cast<OpKernelStateWrapper<std::pair<size_t, size_t>>*>(
      user_kernel->GetOpKernelState().get());
  CHECK_NOTNULL(state);
  state->Mutable()->first = act_num_cnt_;
  state->Mutable()->second = total_pack_num_;
  AsyncLaunchKernel();
  act_num_cnt_ += 1;
}

void PackActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (act_num_cnt_ == total_pack_num_) { HandleProducedNaiveDataRegstToConsumer(); }
}

void PackActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  HandleConsumedNaiveDataRegstToProducer();
  if (act_num_cnt_ == total_pack_num_) { act_num_cnt_ = 0; }
}

REGISTER_ACTOR(TaskType::kPack, PackActor);

}  // namespace oneflow
