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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

class VariableInplaceBufferActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(VariableInplaceBufferActor);
  VariableInplaceBufferActor() : buffer_count_(0), buffer_size_(0), need_all_regst_return_(false){};
  ~VariableInplaceBufferActor() override = default;

 private:
  // NOTE(chengcheng): Empty rs for naive and inplace regst, all regst is customized.
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedProducedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  void TakeOverInplaceConsumedAndProduced(
      const PbMap<std::string, RegstDescProto>& produced_ids) override {
    // NOTE(chengcheng): all regst is customized.
    inplace_consumed_rs_.InitedDone();
    inplace_produced_rs_.InitedDone();
  }

  // NOTE(chengcheng): naive ready for read/write
  bool IsCustomizedReadReady() const override {
    bool is_ready_ready = (!need_all_regst_return_) && consumed_var_rs_.IsCurSlotReady();
    LOG(INFO) << " ccActorLog: actor: " << actor_id() << " is_ready_ready: " << is_ready_ready
              << " of need_all_regst_return_ = " << need_all_regst_return_
              << " consumed_var_rs_.IsCurSlotReady = " << consumed_var_rs_.IsCurSlotReady();
    return (!need_all_regst_return_) && consumed_var_rs_.IsCurSlotReady();
  }
  bool IsCustomizedWriteReady() const override {
    LOG(INFO) << " ccActorLog: actor: " << actor_id()
              << " is_write_ready: " << produced_buffer_rs_.IsCurSlotReady();
    return produced_buffer_rs_.IsCurSlotReady();
  }
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override {}
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override {
    // all Messages are flushed
    return ReceiveEordMsg(consumed_var_regst_desc_id_);
  }

  void VirtualActorInit(const TaskProto& task_proto) override;
  void Act() override;

  void UpdtStateAsCustomizedProducedRegst(Regst* regst) override {
    CHECK_EQ(regst->regst_desc_id(), produced_buffer_regst_desc_id_);
    CHECK_EQ(produced_buffer_rs_.TryPushBackRegst(regst), 0);
    LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << buffer_count_
              << " regst_desc_id: " << produced_buffer_regst_desc_id_ << " ready size = "
              << produced_buffer_rs_.GetReadyRegstSize(produced_buffer_regst_desc_id_);
    if (need_all_regst_return_
        && produced_buffer_rs_.GetReadyRegstSize(produced_buffer_regst_desc_id_) == buffer_size_) {
      Regst* in_regst = consumed_var_rs_.Front(consumed_var_regst_desc_id_);
      CHECK(in_regst);
      AsyncSendRegstMsgToProducer(in_regst);
      CHECK_EQ(0, consumed_var_rs_.TryPopFrontRegst(consumed_var_regst_desc_id_));
      need_all_regst_return_ = false;

      LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << buffer_count_
                << " consumed_regst_desc_id: " << consumed_var_regst_desc_id_
                << " return with all produced regst.";
    }
  }

  void AsyncSendCustomizedProducedRegstMsgToConsumer() override {
    CHECK(consumed_var_rs_.IsCurSlotReady());
    CHECK(produced_buffer_rs_.IsCurSlotReady());
    Regst* const buffer_regst = produced_buffer_rs_.Front(produced_buffer_regst_desc_id_);
    CHECK_GT(HandleRegstToConsumer(buffer_regst), 0);
    produced_buffer_rs_.PopFrontRegsts({produced_buffer_regst_desc_id_});

    LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << buffer_count_
              << " Send buffer regst " << produced_buffer_regst_desc_id_ << " to Consumer.";
  }

  void AsyncSendCustomizedConsumedRegstMsgToProducer() override {
    if (!need_all_regst_return_) {
      Regst* const var_regst = consumed_var_rs_.Front(consumed_var_regst_desc_id_);
      CHECK_NOTNULL(var_regst);
      AsyncSendRegstMsgToProducer(var_regst);
      CHECK_EQ(0, consumed_var_rs_.TryPopFrontRegst(consumed_var_regst_desc_id_));

      LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << buffer_count_
                << " return var regst " << consumed_var_regst_desc_id_ << " to producer.";
    } else {
      LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << buffer_count_
                << " NOT return var regst for waiting inplace buffer regst returned. ";
    }
  }

  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) override {
    if (var_regst_ == nullptr) {
      var_regst_ = msg.regst();
    } else {
      CHECK_EQ(var_regst_, msg.regst());
    }
    CHECK_EQ(0, consumed_var_rs_.TryPushBackRegst(var_regst_));
    LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << buffer_count_
              << " receive var regst: " << var_regst_->regst_desc_id();
  }

  // NOTE(chengcheng) buffer_count_ start from 1 for % == 0 at n - 1 iter to sync.
  int64_t buffer_count_;
  int64_t buffer_size_;
  bool need_all_regst_return_;
  // input var
  int64_t consumed_var_regst_desc_id_;
  RegstSlot consumed_var_rs_;
  Regst* var_regst_;

  // output buffer
  int64_t produced_buffer_regst_desc_id_;
  RegstSlot produced_buffer_rs_;
};

void VariableInplaceBufferActor::VirtualActorInit(const TaskProto& proto) {
  const OperatorConf op_conf =
      proto.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
  buffer_size_ = user_op::UserOpConfWrapper(op_conf).attr<int64_t>("buffer_size");
  CHECK_GT(buffer_size_, 1);

  // input
  const auto& consumed_ids = proto.consumed_regst_desc_id();
  CHECK_EQ(consumed_ids.size(), 1);
  CHECK(consumed_ids.find("in") != consumed_ids.end());
  const auto& in_ids = consumed_ids.at("in");
  CHECK_EQ(in_ids.regst_desc_id_size(), 1);
  consumed_var_regst_desc_id_ = in_ids.regst_desc_id(0);
  consumed_var_rs_.InsertRegstDescId(consumed_var_regst_desc_id_);
  consumed_var_rs_.InitedDone();
  var_regst_ = nullptr;

  // output
  const auto& produced_ids = proto.produced_regst_desc();
  CHECK_EQ(produced_ids.size(), 1);
  CHECK(produced_ids.find("out") != produced_ids.end());
  const RegstDescProto& out_regst_desc = produced_ids.at("out");
  CHECK(!out_regst_desc.enable_reuse_mem());
  CHECK_EQ(out_regst_desc.register_num(), 1);
  // check inplace
  CHECK_EQ(out_regst_desc.inplace_consumed_regst_desc_id(), consumed_var_regst_desc_id_);
  produced_buffer_regst_desc_id_ = out_regst_desc.regst_desc_id();
  produced_buffer_rs_.InsertRegstDescId(produced_buffer_regst_desc_id_);
  produced_buffer_rs_.InitedDone();
  // Regst number hacking
  for (int64_t i = 1; i < buffer_size_; ++i) {
    Global<RegstMgr>::Get()->NewRegsts(out_regst_desc, [this](Regst* regst) {
      produced_regsts_[this->produced_buffer_regst_desc_id_].emplace_back(regst);
      produced_regst2reading_cnt_[regst] = 0;
    });
  }

  ForEachProducedRegst([&](Regst* regst) {
    if (regst->regst_desc_id() != produced_buffer_regst_desc_id_) { return; }
    CHECK_EQ(0, produced_buffer_rs_.TryPushBackRegst(regst));
  });

  LOG(INFO) << " ccActorLog: actor: " << actor_id() << " has produced_buffer_rs_ regst_descs = "
            << produced_buffer_rs_.total_regst_desc_cnt() << " with regsts size = "
            << produced_buffer_rs_.GetReadyRegstSize(produced_buffer_regst_desc_id_);
  LOG(INFO) << " ccActorLog: actor: " << actor_id()
            << " has consumed_var_rs_ regst_descs = " << consumed_var_rs_.total_regst_desc_cnt()
            << " with regsts size = "
            << consumed_var_rs_.GetReadyRegstSize(consumed_var_regst_desc_id_);
  LOG(INFO)
      << " ccActorLog: actor: " << actor_id()
      << " has inplace_consumed_rs_ regst_descs = " << inplace_consumed_rs_.total_regst_desc_cnt()
      << " \nhas inplace_produced_rs_ regst_descs = " << inplace_produced_rs_.total_regst_desc_cnt()
      << " \nhas naive_consumed_rs_ regst_descs = " << naive_consumed_rs_.total_regst_desc_cnt()
      << " \nhas naive_produced_rs_ regst_descs = " << naive_produced_rs_.total_regst_desc_cnt();

  // LOG(WARNING) << "cclog: VariableInplaceBufferActor init " << proto.DebugString();
  OF_SET_MSG_HANDLER(&VariableInplaceBufferActor::HandlerNormal);
}

void VariableInplaceBufferActor::Act() {
  // NOTE(chengcheng): add count before act.
  buffer_count_ += 1;
  LOG(INFO) << " ccActorLog: actor: " << actor_id() << " buffer_count_ = " << buffer_count_
            << " buffer_size = " << buffer_size_;
  if (buffer_count_ >= buffer_size_) {
    CHECK(!need_all_regst_return_);
    LOG(INFO) << " ccActorLog: actor: " << actor_id() << " need_all_regst_return = true";
    need_all_regst_return_ = true;
  }
  buffer_count_ = buffer_count_ % buffer_size_;
  // NOTE(chengcheng):
  //   Variable Inplace Buffer Actor using inplace input with all buffer_size_ num output regst,
  //   so Act() will Do Nothing.
  Regst* out_regst = produced_buffer_rs_.Front(produced_buffer_regst_desc_id_);
  Regst* in_regst = consumed_var_rs_.Front(consumed_var_regst_desc_id_);
  CHECK(out_regst && in_regst);
  // LOG(WARNING) << "cclog: VariableInplaceBufferActor: "
  //             << out_regst->regst_desc()->regst_desc_type().DebugString();
  CHECK(out_regst->main_mem_ptr() == in_regst->main_mem_ptr());
  CHECK(out_regst->separated_header_mem_ptr() == in_regst->separated_header_mem_ptr());
  CHECK_EQ(out_regst->regst_desc()->MainByteSize4OneRegst(),
           in_regst->regst_desc()->MainByteSize4OneRegst());
  CHECK_EQ(out_regst->regst_desc()->SeparatedHeaderByteSize4OneRegst(),
           in_regst->regst_desc()->SeparatedHeaderByteSize4OneRegst());
}

REGISTER_ACTOR(TaskType::kVariableInplaceBuffer, VariableInplaceBufferActor);

}  // namespace oneflow
