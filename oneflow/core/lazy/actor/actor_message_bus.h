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
#ifndef ONEFLOW_CORE_LAZY_ACTOR_ACTOR_MESSAGE_BUS_H_
#define ONEFLOW_CORE_LAZY_ACTOR_ACTOR_MESSAGE_BUS_H_

#include "oneflow/core/lazy/actor/actor_message.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class ActorMsgBus final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorMsgBus);
  ~ActorMsgBus() = default;

  void SendMsg(const ActorMsg& msg);
  void SendMsgWithoutCommNet(const ActorMsg& msg);
  void SendMsgsWithoutCommNet(const ActorMsg* msgs, size_t n, int64_t thrd_id);

 private:
  friend class Singleton<ActorMsgBus>;
  ActorMsgBus() = default;
  HashMap<std::pair<int64_t, int64_t>, int64_t>
      regst_desc_id_dst_actor_id2comm_net_sequence_number_;
  std::mutex regst_desc_id_dst_actor_id2comm_net_sequence_number_mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_ACTOR_ACTOR_MESSAGE_BUS_H_
