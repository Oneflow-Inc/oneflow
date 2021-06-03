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
#ifndef ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

enum class ActorCmd {
  kInitModel = 0,     // MdUpdt Actor
  kSendInitialModel,  // MdUpdt Actor
  kStart,             // Source Actor
  kStopThread,
  kConstructActor
};

enum class ActorMsgType { kRegstMsg = 0, kEordMsg, kCmdMsg };

class ActorMsg final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ActorMsg);
  ActorMsg() = default;
  ~ActorMsg() = default;

  // Build Msg
  static ActorMsg BuildRegstMsgToConsumer(int64_t producer, int64_t consumer, Regst*);
  static ActorMsg BuildRegstMsgToProducer(int64_t consumer, int64_t producer, Regst*);
  static ActorMsg BuildEordMsg(int64_t consumer, int64_t regst_desc_id);
  static ActorMsg BuildCommandMsg(int64_t dst_actor_id, ActorCmd cmd);

  // Getters
  int64_t SrcMachineId() const;
  int64_t src_actor_id() const { return src_actor_id_; }
  int64_t dst_actor_id() const { return dst_actor_id_; }
  ActorMsgType msg_type() const { return msg_type_; }
  ActorCmd actor_cmd() const;
  Regst* regst() const;
  int64_t regst_desc_id() const;
  int64_t piece_id() const;
  int64_t act_id() const;
  void* comm_net_token() const;
  bool has_sole_empty_blob() const;
  int64_t eord_regst_desc_id() const;

  // Serialize
  template<typename StreamT>
  void Serialize(StreamT& out_stream) const {
    out_stream.Write(this, sizeof(ActorMsg));
  }
  template<typename StreamT>
  void Deserialize(StreamT& in_stream) {
    in_stream.Read(this, sizeof(ActorMsg));
  }

 private:
  struct RegstWrapper {
    Regst* regst;
    void* comm_net_token;
    RegstStatus regst_status;
    bool has_sole_empty_blob;
  };

  int64_t src_actor_id_;
  int64_t dst_actor_id_;
  ActorMsgType msg_type_;
  union {
    ActorCmd actor_cmd_;
    RegstWrapper regst_wrapper_;
    int64_t eord_regst_desc_id_;
  };
};

template<typename StreamT>
StreamT& operator<<(StreamT& out_stream, const ActorMsg& msg) {
  msg.Serialize(out_stream);
}

template<typename StreamT>
StreamT& operator>>(StreamT& in_stream, const ActorMsg& msg) {
  msg.Deserialize(in_stream);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
