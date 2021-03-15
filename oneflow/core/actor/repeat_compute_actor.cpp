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

namespace oneflow {

class RepeatCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatCompActor);
  RepeatCompActor() = default;
  ~RepeatCompActor() override = default;

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

void RepeatCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& in_time_shape = Global<RegstMgr>::Get()
                                   ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                   .data_regst_time_shape();
  const Shape& out_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                    .data_regst_time_shape();
  CHECK_GE(out_time_shape.NumAxes(), 1);
  CHECK_EQ(in_time_shape.NumAxes() + 1, out_time_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, in_time_shape.NumAxes()) {
    CHECK_EQ(in_time_shape.At(i), out_time_shape.At(i));
  }
  repeat_num_ = out_time_shape.At(out_time_shape.NumAxes() - 1);
  repeat_count_ = 0;
  cur_piece_id_ = 0;

  const RegstDescProto& out_regst_desc = proto.produced_regst_desc().at("out");
  CHECK(!out_regst_desc.enable_reuse_mem());
  CHECK_EQ(out_regst_desc.register_num(), 1);
  OF_SET_MSG_HANDLER(&RepeatCompActor::HandlerNormal);
}

void RepeatCompActor::Act() {
  // reset repeat_count if need
  if (repeat_count_ == repeat_num_) { repeat_count_ = 0; }

  if (repeat_count_ == 0) {
    KernelCtx kernel_ctx = GenDefaultKernelCtx();
    AsyncLaunchKernel(kernel_ctx);
  }

  repeat_count_ += 1;
}

void RepeatCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (repeat_count_ == repeat_num_) {
    HandleConsumedNaiveDataRegstToProducer([](Regst* regst) { return true; });
  }
}

void RepeatCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
  cur_piece_id_ += 1;
}

bool RepeatCompActor::ConsumedCtrlRegstValid(int64_t regst_desc_id) const {
  return repeat_count_ == repeat_num_;
}

REGISTER_ACTOR(TaskType::kRepeat, RepeatCompActor);

}  // namespace oneflow
