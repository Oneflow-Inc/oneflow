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
#ifndef ONEFLOW_CORE_ACTOR_REPEAT_FORWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REPEAT_FORWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class RepeatForwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatForwardCompActor);
  RepeatForwardCompActor() = default;
  ~RepeatForwardCompActor() override = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void Act() override;
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  bool ConsumedCtrlRegstValid(int64_t regst_desc_id) const override;

  int64_t repeat_num_;
  int64_t repeat_count_;
  int64_t cur_piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REPEAT_FORWARD_COMPUTE_ACTOR_H_
