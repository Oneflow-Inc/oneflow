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
#include "oneflow/core/actor/compute_actor.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

class SspVariableProxyCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SspVariableProxyCompActor);
  SspVariableProxyCompActor() = default;
  ~SspVariableProxyCompActor() override = default;

 protected:
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedProducedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  bool CheckOutputActId(int64_t regst_desc_id) const override { return false; }
  bool IsCustomizedReadReady() const override { return consumed_var_rs_.IsCurSlotReady(); }
  bool IsCustomizedWriteReady() const override {
    int64_t cur_staleness = (received_var_piece_id_ - ack_msg_returned_ref_piece_id_);
    return ((cur_staleness <= staleness() /* bounded staleness */)
            && (produced_value_rs_.IsCurSlotReady()
                /* able to send messages to consumers of output `value` */))
           || (produced_ref_rs_.IsCurSlotReady()
               /* able to send or to flush messages to consumers of output `ref` */);
  }
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override {}
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override {
    // all Messages are flushed
    return ReceiveEordMsg(consumed_var_regst_desc_id_)
           && (received_var_piece_id_ <= ack_msg_returned_value_piece_id_ + 1
               /* there is no need to wait the last piece */)
           && (received_var_piece_id_ == ack_msg_returned_ref_piece_id_);
  }

  void UpdtStateAsCustomizedProducedRegst(Regst* regst) override {
    if (regst->regst_desc_id() == produced_value_regst_desc_id_) {
      ++ack_msg_returned_value_piece_id_;
      CHECK_EQ(regst->piece_id(), ack_msg_returned_value_piece_id_);
      CHECK_EQ(regst, GetRingBufferValueRegst(ack_msg_returned_value_piece_id_));
      CHECK_EQ(0, produced_value_rs_.TryPushBackRegst(regst));
      if (ack_msg_returned_ref_piece_id_ == ack_msg_returned_value_piece_id_
          /* All mutable consumers to ref regst has done their job */) {
        // The updated ref regst are not synced into value regst yet.
        SyncRefRegstIntoValueRegst(ack_msg_returned_value_piece_id_);
      } else if (ack_msg_returned_ref_piece_id_ > ack_msg_returned_value_piece_id_) {
        // The ACK of ref resgt can just be slightly earlier than the one of value regst.
        // `slightly` means `ack_msg_returned_ref_piece_id_ == ack_msg_returned_value_piece_id_`
        UNIMPLEMENTED();
      } else {
        // Do nothing. The ref data is not updated yet.
      }
    } else if (regst->regst_desc_id() == produced_ref_regst_desc_id_) {
      ++ack_msg_returned_ref_piece_id_;
      CHECK_EQ(regst->piece_id(), ack_msg_returned_ref_piece_id_);
      CHECK_EQ(regst, ref_regst_);
      if (ack_msg_returned_value_piece_id_ >= ack_msg_returned_ref_piece_id_
          /* All const consumers to value regst has done their job */) {
        SyncRefRegstIntoValueRegst(ack_msg_returned_ref_piece_id_);
      } else {
        // Do nothing. The ACK of value regst will do the sync work
      }
    } else {
      UNIMPLEMENTED();
    }
  }

  void AsyncSendCustomizedProducedRegstMsgToConsumer() override {
    if (consumed_var_rs_.IsCurSlotReady() && produced_value_rs_.IsCurSlotReady()) {
      Regst* const value_regst = produced_value_rs_.Front(produced_value_regst_desc_id_);
      if (value_regst->consumers_actor_id().empty()) {
        ++ack_msg_returned_value_piece_id_;
      } else {
        Regst* const var_regst = consumed_var_rs_.Front(consumed_var_regst_desc_id_);
        CHECK_EQ(received_var_piece_id_, var_regst->piece_id());
        CHECK_EQ(value_regst, GetRingBufferValueRegst(received_var_piece_id_));
        value_regst->set_piece_id(received_var_piece_id_);
        CHECK_GT(HandleRegstToConsumer(value_regst, [](int64_t) { return true; }), 0);
        produced_value_rs_.PopFrontRegsts({produced_value_regst_desc_id_});
      }
    }
    if ((ack_msg_returned_ref_piece_id_ < received_var_piece_id_)
        && produced_ref_rs_.IsCurSlotReady()) {
      Regst* const ref_regst = produced_ref_rs_.Front(produced_ref_regst_desc_id_);
      if (ref_regst->consumers_actor_id().empty()) {
        ++ack_msg_returned_ref_piece_id_;
      } else {
        ref_regst->set_piece_id(ack_msg_returned_ref_piece_id_ + 1);
        CHECK_GT(HandleRegstToConsumer(ref_regst, [](int64_t) { return true; }), 0);
        produced_ref_rs_.PopFrontRegsts({produced_ref_regst_desc_id_});
      }
    }
  }

  void AsyncSendCustomizedConsumedRegstMsgToProducer() override {
    Regst* const var_regst = consumed_var_rs_.Front(consumed_var_regst_desc_id_);
    CHECK_NOTNULL(var_regst);
    AsyncSendRegstMsgToProducer(var_regst);
    CHECK_EQ(0, consumed_var_rs_.TryPopFrontRegst(consumed_var_regst_desc_id_));
  }

  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)> Handler) const override {
    Handler(consumed_var_rs_.Front(consumed_var_regst_desc_id_));
  }

  void TakeOverInplaceConsumedAndProduced(
      const PbMap<std::string, RegstDescProto>& produced_ids) override {
    inplace_consumed_rs_.InitedDone();
    inplace_produced_rs_.InitedDone();
  }

  void VirtualCompActorInit(const TaskProto& task_proto) override {
    CheckInplaceBetweenVarAndRef(task_proto);
    TakeOverVarRegst(task_proto.consumed_regst_desc_id());
    TakeOverRefRegst(task_proto.produced_regst_desc());
    TakeOverValueRegst(task_proto.produced_regst_desc());
    OF_SET_MSG_HANDLER(&SspVariableProxyCompActor::HandlerNormal);
  }

  bool ProducedCtrlRegstValid(int64_t regst_desc_id) const override { return true; }

  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) override {
    if (var_regst_ == nullptr) {
      var_regst_ = msg.regst();
    } else {
      CHECK_EQ(var_regst_, msg.regst());
    }
    CHECK_EQ(0, consumed_var_rs_.TryPushBackRegst(var_regst_));
    ++received_var_piece_id_;
    CHECK_EQ(var_regst_->piece_id(), received_var_piece_id_);
  }

 private:
  void Act() override {
    if (received_var_piece_id_ == 0) {
      // Initialize all value regsts
      for (int64_t piece_id = 0; piece_id < staleness(); ++piece_id) {
        CopyRefToValue(GetRingBufferValueRegst(piece_id));
      }
    } else {
      // Do nothing, value regsts are updated in UpdtStateAsCustomizedProducedRegst
    }
  }

  void CheckInplaceBetweenVarAndRef(const TaskProto& task_proto) const {
    int64_t var_id = task_proto.consumed_regst_desc_id().at("var").regst_desc_id(0);
    const auto& ref_regst_desc_proto = task_proto.produced_regst_desc().at("ref");
    CHECK_EQ(ref_regst_desc_proto.inplace_consumed_regst_desc_id(), var_id);
  }

  void TakeOverVarRegst(const PbMap<std::string, RegstDescIdSet>& consumed_ids) {
    received_var_piece_id_ = -1;
    consumed_var_regst_desc_id_ = consumed_ids.at("var").regst_desc_id(0);
    consumed_var_rs_.InsertRegstDescId(consumed_var_regst_desc_id_);
    consumed_var_rs_.InitedDone();
    var_regst_ = nullptr;
  }

  void TakeOverRefRegst(const PbMap<std::string, RegstDescProto>& produced_ids) {
    ack_msg_returned_ref_piece_id_ = -1;
    produced_ref_regst_desc_id_ = produced_ids.at("ref").regst_desc_id();
    produced_ref_rs_.InsertRegstDescId(produced_ref_regst_desc_id_);
    produced_ref_rs_.InitedDone();
    ref_regst_ = nullptr;
    ForEachProducedRegst([&](Regst* regst) {
      if (regst->regst_desc_id() != produced_ref_regst_desc_id_) { return; }
      CHECK(ref_regst_ == nullptr) << "regst_num of ref_regst must equal 1";
      CHECK_EQ(0, produced_ref_rs_.TryPushBackRegst(regst));
      ref_regst_ = regst;
    });
  }

  void TakeOverValueRegst(const PbMap<std::string, RegstDescProto>& produced_ids) {
    ack_msg_returned_value_piece_id_ = -1;
    produced_value_regst_desc_id_ = produced_ids.at("value").regst_desc_id();
    produced_value_rs_.InsertRegstDescId(produced_value_regst_desc_id_);
    produced_value_rs_.InitedDone();
    ForEachProducedRegst([&](Regst* regst) {
      if (regst->regst_desc_id() != produced_value_regst_desc_id_) { return; }
      CHECK_EQ(0, produced_value_rs_.TryPushBackRegst(regst));
      value_regst_ring_buffer_.push_back(regst);
    });
  }

  void SyncRefRegstIntoValueRegst(int64_t released_piece_id) {
    CopyRefToValue(GetRingBufferValueRegst(released_piece_id));
    CHECK_EQ(0, produced_ref_rs_.TryPushBackRegst(ref_regst_));
  }

  void CopyRefToValue(Regst* value_regst) {
    KernelCtx kernel_ctx = GenDefaultKernelCtx();
    AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
      if (regst_desc_id == consumed_var_regst_desc_id_) {
        return var_regst_;
      } else if (regst_desc_id == produced_ref_regst_desc_id_) {
        return ref_regst_;
      } else if (regst_desc_id == produced_value_regst_desc_id_) {
        return value_regst;
      } else {
        UNIMPLEMENTED();
      }
    });
  }

  Regst* GetRingBufferValueRegst(int64_t value_piece_id) const {
    return value_regst_ring_buffer_.at(value_piece_id % staleness());
  }

  size_t staleness() const { return value_regst_ring_buffer_.size(); }

  // input var
  int64_t received_var_piece_id_;
  int64_t consumed_var_regst_desc_id_;
  RegstSlot consumed_var_rs_;
  Regst* var_regst_;
  // output ref
  // consumers has used the ref regst
  int64_t ack_msg_returned_ref_piece_id_;
  int64_t produced_ref_regst_desc_id_;
  RegstSlot produced_ref_rs_;
  Regst* ref_regst_;
  // output value
  // consumers has used the value regst
  int64_t ack_msg_returned_value_piece_id_;
  int64_t produced_value_regst_desc_id_;
  RegstSlot produced_value_rs_;
  std::vector<Regst*> value_regst_ring_buffer_;
};

REGISTER_ACTOR(TaskType::kSspVariableProxy, SspVariableProxyCompActor);

}  // namespace oneflow
