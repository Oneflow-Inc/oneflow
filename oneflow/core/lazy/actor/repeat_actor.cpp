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

namespace oneflow {

class RepeatActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatActor);
  RepeatActor() = default;
  ~RepeatActor() override = default;

 private:
  void VirtualActorInit(const TaskProto& proto) override;
  void Act() override;
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  bool ConsumedCtrlRegstValid(int64_t regst_desc_id) const override;
  bool IsCustomizedWriteReady() const override;

  int64_t repeat_num_;
  int64_t repeat_count_;
};

void RepeatActor::VirtualActorInit(const TaskProto& proto) {
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
  repeat_num_ = out_time_shape.At(out_time_shape.NumAxes() - 1);
  repeat_count_ = 0;
  const RegstDescProto& out_regst_desc = proto.produced_regst_desc().at("out");
  CHECK(!out_regst_desc.enable_reuse_mem());
  CHECK_EQ(out_regst_desc.register_num(), 1);

  // Regst number hacking
  if (naive_consumed_rs_.total_regst_desc_cnt() != 1) {
    LOG(WARNING)
        << "RepeatActor has more than one consumed register. This will impact performance.";
  }

  for (const auto& pair : proto.produced_regst_desc()) {
    const RegstDescProto& regst_desc = pair.second;
    int64_t regst_desc_id = regst_desc.regst_desc_id();
    // This itor begins from 1 because first regst was already inserted in TakeOverNaiveProduced
    for (int64_t i = 1; i < repeat_num_; ++i) {
      Singleton<RegstMgr>::Get()->NewRegsts(regst_desc, [this, regst_desc_id](Regst* regst) {
        produced_regsts_[regst_desc_id].emplace_back(regst);
        produced_regst2reading_cnt_[regst] = 0;
        naive_produced_rs_.TryPushBackRegst(regst);
      });
    }
  }

  OF_SET_MSG_HANDLER(&RepeatActor::HandlerNormal);
}

void RepeatActor::Act() {
  // reset repeat_count if need
  if (repeat_count_ == repeat_num_) { repeat_count_ = 0; }

  if (repeat_count_ == 0) { AsyncLaunchKernel(); }

  repeat_count_ += 1;
}

void RepeatActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (repeat_count_ == repeat_num_) { HandleConsumedNaiveDataRegstToProducer(); }
}

void RepeatActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer();
}

bool RepeatActor::ConsumedCtrlRegstValid(int64_t regst_desc_id) const {
  return repeat_count_ == repeat_num_;
}

bool RepeatActor::IsCustomizedWriteReady() const {
  if (repeat_count_ % repeat_num_ == 0) {
    return total_reading_cnt_ == 0;
  } else {
    return true;
  }
}

REGISTER_ACTOR(TaskType::kRepeat, RepeatActor);

}  // namespace oneflow
