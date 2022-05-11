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

class ConstantInplaceBufferActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantInplaceBufferActor);
  ConstantInplaceBufferActor() : buffer_size_(0){};
  ~ConstantInplaceBufferActor() override = default;

 private:
  void VirtualActorInit(const TaskProto& proto) override;
  void Act() override;

  int32_t buffer_size_;
};

void ConstantInplaceBufferActor::VirtualActorInit(const TaskProto& proto) {
  const OperatorConf op_conf =
      proto.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
  buffer_size_ = user_op::UserOpConfWrapper(op_conf).attr<int64_t>("buffer_size");

  const RegstDescProto& out_regst_desc = proto.produced_regst_desc().at("out");
  CHECK(!out_regst_desc.enable_reuse_mem());
  CHECK_EQ(out_regst_desc.register_num(), 1);

  // Regst number hacking
  const int64_t out_regst_desc_id = out_regst_desc.regst_desc_id();
  for (int64_t i = 1; i < buffer_size_; ++i) {
    Global<RegstMgr>::Get()->NewRegsts(out_regst_desc, [this, out_regst_desc_id](Regst* regst) {
      produced_regsts_[out_regst_desc_id].emplace_back(regst);
      produced_regst2reading_cnt_[regst] = 0;
      inplace_produced_rs_.TryPushBackRegst(regst);
    });
  }
  LOG(WARNIG) << "cclog: ConstantInplaceBufferActor init " << proto.DebugString();
  OF_SET_MSG_HANDLER(&ConstantInplaceBufferActor::HandlerNormal);
}

void ConstantInplaceBufferActor::Act() {
  // NOTE(chengcheng):
  //   Constant Inplace Buffer Actor using inplace input with all buffer_size_ num output regst,
  //   so Act() will Do Nothing.
  Regst* out_regst = GetNaiveCurWriteable("out");
  Regst* in_regst = GetNaiveCurReadable("in");
  LOG(WARNING) << "cclog: ConstantInplaceBufferActor: "
               << out_regst->regst_desc()->regst_desc_type().DebugString();
  CHECK(out_regst->main_mem_ptr() == in_regst->main_mem_ptr());
  CHECK(out_regst->separated_header_mem_ptr() == in_regst->separated_header_mem_ptr());
  CHECK_EQ(out_regst->regst_desc()->MainByteSize4OneRegst(),
           in_regst->regst_desc()->MainByteSize4OneRegst());
  CHECK_EQ(out_regst->regst_desc()->SeparatedHeaderByteSize4OneRegst(),
           in_regst->regst_desc()->SeparatedHeaderByteSize4OneRegst());
}

REGISTER_ACTOR(TaskType::kConstantInplaceBuffer, ConstantInplaceBufferActor);

}  // namespace oneflow
