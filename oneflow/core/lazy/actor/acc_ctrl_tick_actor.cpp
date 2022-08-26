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

class AccCtrlTickActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccCtrlTickActor);
  AccCtrlTickActor()
      : acc_cnt_(0),
        max_acc_num_(0),
        inplace_consume_(false),
        consumed_tick_regst_desc_id_(-1),
        produced_tick_regst_desc_id_(-1){};
  virtual ~AccCtrlTickActor() = default;

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

  bool IsCustomizedReadReady() const override {
    bool is_ready_ready = (!inplace_consume_) && consumed_tick_rs_.IsCurSlotReady();
    LOG(INFO) << " ccActorLog: actor: " << actor_id() << " is_ready_ready: " << is_ready_ready
              << " of inplace_consume_ = " << inplace_consume_
              << " consumed_tick_rs_.IsCurSlotReady = " << consumed_tick_rs_.IsCurSlotReady();
    return (!inplace_consume_) && consumed_tick_rs_.IsCurSlotReady();
  }
  bool IsCustomizedWriteReady() const override {
    LOG(INFO) << " ccActorLog: actor: " << actor_id()
              << " is_write_ready: " << produced_tick_rs_.IsCurSlotReady();
    return produced_tick_rs_.IsCurSlotReady();
  }

  void NormalProcessCustomizedEordMsg(const ActorMsg&) override {}
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override {
    // all Messages are flushed
    return ReceiveEordMsg(consumed_tick_regst_desc_id_);
  }

  void VirtualActorInit(const TaskProto& proto) override;
  void Act() override;
  void AsyncSendCustomizedProducedRegstMsgToConsumer() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;
  void UpdtStateAsCustomizedProducedRegst(Regst* regst) override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) override;

  int32_t acc_cnt_;
  int32_t max_acc_num_;
  bool inplace_consume_;
  int64_t consumed_tick_regst_desc_id_;
  int64_t produced_tick_regst_desc_id_;
  RegstSlot consumed_tick_rs_;
  RegstSlot produced_tick_rs_;
};

void AccCtrlTickActor::VirtualActorInit(const TaskProto& proto) {
  acc_cnt_ = 0;
  // const OperatorConf op_conf =
  //    proto.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
  // max_acc_num_ = user_op::UserOpConfWrapper(op_conf).attr<int32_t>("max_acc_num");

  // NOTE(chengcheng): check time shape equal max_acc_num
  const Shape& in_time_shape = Singleton<RegstMgr>::Get()
                                   ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                   .data_regst_time_shape();
  max_acc_num_ = in_time_shape.elem_cnt();
  CHECK_GT(max_acc_num_, 1);

  /*
  const Shape& out_time_shape = Singleton<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                    .data_regst_time_shape();
  CHECK_EQ(in_time_shape.elem_cnt() % out_time_shape.elem_cnt(), 0);
  CHECK_EQ(in_time_shape.elem_cnt() / out_time_shape.elem_cnt(), max_acc_num_);
  CHECK_GT(max_acc_num_, 1);
  */

  // input
  const auto& consumed_ids = proto.consumed_regst_desc_id();
  CHECK_EQ(consumed_ids.size(), 1);
  CHECK(consumed_ids.find("in") != consumed_ids.end());
  const auto& in_ids = consumed_ids.at("in");
  CHECK_EQ(in_ids.regst_desc_id_size(), 1);
  consumed_tick_regst_desc_id_ = in_ids.regst_desc_id(0);
  consumed_tick_rs_.InsertRegstDescId(consumed_tick_regst_desc_id_);
  consumed_tick_rs_.InitedDone();

  // output
  CHECK_EQ(proto.produced_regst_desc().size(), 1);
  for (const auto& pair : proto.produced_regst_desc()) {
    const RegstDescProto& out_regst_desc = pair.second;
    if (out_regst_desc.regst_desc_type().has_ctrl_regst_desc()) {
      CHECK_EQ(out_regst_desc.register_num(), 1);
      CHECK_EQ(produced_tick_regst_desc_id_, -1);
      produced_tick_regst_desc_id_ = out_regst_desc.regst_desc_id();
      produced_tick_rs_.InsertRegstDescId(produced_tick_regst_desc_id_);
      produced_tick_rs_.InitedDone();
    }
  }
  CHECK_NE(produced_tick_regst_desc_id_, -1);

  ForEachProducedRegst([&](Regst* regst) {
    // if (regst->regst_desc_id() == produced_tick_regst_desc_id_) {
    CHECK_EQ(regst->regst_desc_id(), produced_tick_regst_desc_id_);
    CHECK_EQ(0, produced_tick_rs_.TryPushBackRegst(regst));
    // }
  });

  LOG(INFO) << " ccActorLog: actor: " << actor_id()
            << " has produced_tick_rs_ regst_descs = " << produced_tick_rs_.total_regst_desc_cnt()
            << " with regsts size = "
            << produced_tick_rs_.GetReadyRegstSize(produced_tick_regst_desc_id_);
  LOG(INFO) << " ccActorLog: actor: " << actor_id()
            << " has consumed_tick_rs_ regst_descs = " << consumed_tick_rs_.total_regst_desc_cnt()
            << " with regsts size = "
            << consumed_tick_rs_.GetReadyRegstSize(consumed_tick_regst_desc_id_);
  LOG(INFO)
      << " ccActorLog: actor: " << actor_id()
      << " has inplace_consumed_rs_ regst_descs = " << inplace_consumed_rs_.total_regst_desc_cnt()
      << " \nhas inplace_produced_rs_ regst_descs = " << inplace_produced_rs_.total_regst_desc_cnt()
      << " \nhas naive_consumed_rs_ regst_descs = " << naive_consumed_rs_.total_regst_desc_cnt()
      << " \nhas naive_produced_rs_ regst_descs = " << naive_produced_rs_.total_regst_desc_cnt();
  OF_SET_MSG_HANDLER(&AccCtrlTickActor::HandlerNormal);
}

void AccCtrlTickActor::Act() {
  acc_cnt_ += 1;
  LOG(INFO) << " ccActorLog: actor: " << actor_id() << " acc_count_ = " << acc_cnt_
            << " max_acc_num = " << max_acc_num_;
  if (acc_cnt_ == max_acc_num_) {
    CHECK(!inplace_consume_);
    inplace_consume_ = true;
    LOG(INFO) << " ccActorLog: actor: " << actor_id() << " inplace_consume_ = true";
    acc_cnt_ = 0;
  }
}

void AccCtrlTickActor::AsyncSendCustomizedProducedRegstMsgToConsumer() {
  if (inplace_consume_) {
    CHECK(consumed_tick_rs_.IsCurSlotReady());  // inplace consume
    CHECK(produced_tick_rs_.IsCurSlotReady());
    Regst* const tick_regst = produced_tick_rs_.Front(produced_tick_regst_desc_id_);
    CHECK_GT(HandleRegstToConsumer(tick_regst), 0);
    produced_tick_rs_.PopFrontRegsts({produced_tick_regst_desc_id_});

    LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << acc_cnt_
              << " Send ctrl_tick regst " << produced_tick_regst_desc_id_ << " to Consumer.";
  } else {
    LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << acc_cnt_
              << " SKIP to send produced to consumer.";
  }
}

void AccCtrlTickActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  if (!inplace_consume_) {
    Regst* const tick_regst = consumed_tick_rs_.Front(consumed_tick_regst_desc_id_);
    CHECK_NOTNULL(tick_regst);
    AsyncSendRegstMsgToProducer(tick_regst);
    CHECK_EQ(0, consumed_tick_rs_.TryPopFrontRegst(consumed_tick_regst_desc_id_));

    LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << acc_cnt_
              << " return tick regst " << consumed_tick_regst_desc_id_ << " to producer.";
  } else {
    LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << acc_cnt_
              << " NOT return tick regst for waiting inplace tick regst returned. ";
  }
}

void AccCtrlTickActor::UpdtStateAsCustomizedProducedRegst(Regst* regst) {
  CHECK(inplace_consume_);
  CHECK_EQ(regst->regst_desc_id(), produced_tick_regst_desc_id_);
  CHECK_EQ(produced_tick_rs_.TryPushBackRegst(regst), 0);
  LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << acc_cnt_
            << " regst_desc_id: " << produced_tick_regst_desc_id_ << " ready size = "
            << produced_tick_rs_.GetReadyRegstSize(produced_tick_regst_desc_id_);

  Regst* in_regst = consumed_tick_rs_.Front(consumed_tick_regst_desc_id_);
  CHECK(in_regst);
  AsyncSendRegstMsgToProducer(in_regst);
  CHECK_EQ(0, consumed_tick_rs_.TryPopFrontRegst(consumed_tick_regst_desc_id_));
  inplace_consume_ = false;

  LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << acc_cnt_
            << " consumed_regst_desc_id: " << consumed_tick_regst_desc_id_
            << " return with all produced regst.";
}

void AccCtrlTickActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_tick_rs_.TryPushBackRegst(msg.regst()));
  LOG(INFO) << "ccActorLog: actor: " << actor_id() << " in count: " << acc_cnt_
            << " receive input regst: " << msg.regst()->regst_desc_id();
}

REGISTER_ACTOR(TaskType::kAccCtrlTick, AccCtrlTickActor);

}  // namespace oneflow
