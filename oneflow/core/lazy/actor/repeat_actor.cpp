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

class RepeatActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatActor);
  RepeatActor()
      : repeat_count_(0),
        repeat_num_(0),
        wait_all_regst_return_(false),
        consumed_var_regst_desc_id_(-1),
        produced_repeat_var_regst_desc_id_(-1){};
  ~RepeatActor() override = default;

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

  bool IsCustomizedReadReady() const override {
    return (!wait_all_regst_return_) && consumed_var_rs_.IsCurSlotReady();
  }
  bool IsCustomizedWriteReady() const override {
    return (!wait_all_regst_return_) && produced_repeat_var_rs_.IsCurSlotReady();
  }

  void NormalProcessCustomizedEordMsg(const ActorMsg&) override {}
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override {
    // all Messages are flushed
    return ReceiveEordMsg(consumed_var_regst_desc_id_);
  }

  void VirtualActorInit(const TaskProto& proto) override;
  void Act() override;
  void AsyncSendCustomizedProducedRegstMsgToConsumer() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;
  void UpdtStateAsCustomizedProducedRegst(Regst* regst) override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) override;

  int32_t repeat_count_;
  int32_t repeat_num_;
  bool wait_all_regst_return_;
  int64_t consumed_var_regst_desc_id_;
  int64_t produced_repeat_var_regst_desc_id_;
  RegstSlot consumed_var_rs_;
  RegstSlot produced_repeat_var_rs_;
};

void RepeatActor::VirtualActorInit(const TaskProto& proto) {
  repeat_count_ = 0;
  const OperatorConf& op_conf =
      proto.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
  repeat_num_ = user_op::UserOpConfWrapper(op_conf).attr<int32_t>("repeat_num");

  const Shape& in_time_shape = Singleton<RegstMgr>::Get()
                                   ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                   .data_regst_time_shape();
  const Shape& out_time_shape = Singleton<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                    .data_regst_time_shape();
  CHECK_GE(out_time_shape.NumAxes(), 1);
  CHECK_EQ(in_time_shape.NumAxes() + 1, out_time_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, in_time_shape.NumAxes()) {
    CHECK_EQ(in_time_shape.At(i), out_time_shape.At(i));
  }
  CHECK_EQ(repeat_num_, out_time_shape.At(out_time_shape.NumAxes() - 1));

  // input
  const auto& consumed_ids = proto.consumed_regst_desc_id();
  auto in_it = consumed_ids.find("in");
  CHECK(in_it != consumed_ids.end());
  CHECK_EQ(in_it->second.regst_desc_id_size(), 1);
  consumed_var_regst_desc_id_ = in_it->second.regst_desc_id(0);
  consumed_var_rs_.InsertRegstDescId(consumed_var_regst_desc_id_);
  consumed_var_rs_.InitedDone();

  // output
  const auto& produced_ids = proto.produced_regst_desc();
  auto out_it = produced_ids.find("out");
  CHECK(out_it != produced_ids.end());
  const RegstDescProto& out_regst_desc = out_it->second;
  CHECK(!out_regst_desc.enable_reuse_mem());
  CHECK_EQ(out_regst_desc.register_num(), 1);
  // check inplace
  CHECK_EQ(out_regst_desc.inplace_consumed_regst_desc_id(), consumed_var_regst_desc_id_);
  produced_repeat_var_regst_desc_id_ = out_regst_desc.regst_desc_id();
  produced_repeat_var_rs_.InsertRegstDescId(produced_repeat_var_regst_desc_id_);
  produced_repeat_var_rs_.InitedDone();

  // NOTE(chengcheng): repeat actor may has output ctrl regst. ctrl regst also need hack regst num.
  for (const auto& pair : proto.produced_regst_desc()) {
    const RegstDescProto& regst_desc = pair.second;
    int64_t regst_desc_id = regst_desc.regst_desc_id();
    // This iter begins from 1 because first ctrl regst was already inserted in
    // TakeOverNaiveProduced
    for (int64_t i = 1; i < repeat_num_; ++i) {
      Singleton<RegstMgr>::Get()->NewRegsts(regst_desc, [this, regst_desc_id](Regst* regst) {
        produced_regsts_[regst_desc_id].emplace_back(regst);
        produced_regst2reading_cnt_[regst] = 0;
        if (regst_desc_id != produced_repeat_var_regst_desc_id_) {
          CHECK_EQ(0, naive_produced_rs_.TryPushBackRegst(regst));
        }
      });
    }
  }

  ForEachProducedRegst([&](Regst* regst) {
    if (regst->regst_desc_id() == produced_repeat_var_regst_desc_id_) {
      CHECK_EQ(0, produced_repeat_var_rs_.TryPushBackRegst(regst));
    }
  });

  for (const auto& pair : proto.produced_regst_desc()) {
    const RegstDescProto& regst_desc = pair.second;
    int64_t regst_desc_id = regst_desc.regst_desc_id();
    if (regst_desc_id == produced_repeat_var_regst_desc_id_) {
      CHECK_EQ(produced_repeat_var_rs_.GetReadyRegstSize(regst_desc_id), repeat_num_);
    } else {
      CHECK_EQ(naive_produced_rs_.GetReadyRegstSize(regst_desc_id), repeat_num_);
    }
  }

  OF_SET_MSG_HANDLER(&RepeatActor::HandlerNormal);
}

void RepeatActor::Act() {
  repeat_count_ += 1;

  if (repeat_count_ == repeat_num_) {
    wait_all_regst_return_ = true;
    repeat_count_ = 0;
  }

  Regst* out_regst = produced_repeat_var_rs_.Front(produced_repeat_var_regst_desc_id_);
  Regst* in_regst = consumed_var_rs_.Front(consumed_var_regst_desc_id_);
  CHECK(out_regst && in_regst);
  CHECK(out_regst->body_mem_ptr() == in_regst->body_mem_ptr());
  CHECK(out_regst->header_mem_ptr() == in_regst->header_mem_ptr());
  CHECK_EQ(out_regst->regst_desc()->MainByteSize4OneRegst(),
           in_regst->regst_desc()->MainByteSize4OneRegst());
  CHECK_EQ(out_regst->regst_desc()->SeparatedHeaderByteSize4OneRegst(),
           in_regst->regst_desc()->SeparatedHeaderByteSize4OneRegst());
}

void RepeatActor::AsyncSendCustomizedProducedRegstMsgToConsumer() {
  CHECK(produced_repeat_var_rs_.IsCurSlotReady());
  Regst* const repeat_var_regst = produced_repeat_var_rs_.Front(produced_repeat_var_regst_desc_id_);
  CHECK_GT(HandleRegstToConsumer(repeat_var_regst), 0);
  produced_repeat_var_rs_.PopFrontRegsts({produced_repeat_var_regst_desc_id_});
}

void RepeatActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  // NOTE(chengcheng): do nothing. consumed var regst will return in inplace done.
}

void RepeatActor::UpdtStateAsCustomizedProducedRegst(Regst* regst) {
  CHECK_EQ(regst->regst_desc_id(), produced_repeat_var_regst_desc_id_);
  CHECK_EQ(produced_repeat_var_rs_.TryPushBackRegst(regst), 0);

  if (wait_all_regst_return_
      && produced_repeat_var_rs_.GetReadyRegstSize(produced_repeat_var_regst_desc_id_)
             == repeat_num_) {
    Regst* in_regst = consumed_var_rs_.Front(consumed_var_regst_desc_id_);
    CHECK(in_regst);
    AsyncSendRegstMsgToProducer(in_regst);
    CHECK_EQ(0, consumed_var_rs_.TryPopFrontRegst(consumed_var_regst_desc_id_));
    wait_all_regst_return_ = false;
  }
}

void RepeatActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_var_rs_.TryPushBackRegst(msg.regst()));
}
REGISTER_ACTOR(TaskType::kRepeat, RepeatActor);

}  // namespace oneflow
