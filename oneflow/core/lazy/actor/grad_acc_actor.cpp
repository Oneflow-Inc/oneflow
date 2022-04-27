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

class GradAccActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GradAccActor);
  GradAccActor() : acc_cnt_(0), acc_num_(0){};
  ~GradAccActor() override = default;

 private:
  void Act() override;

  void VirtualActorInit(const TaskProto& proto) override;

  int32_t acc_cnt_;
  int32_t acc_num_;
};

void GradAccActor::VirtualActorInit(const TaskProto& proto) {
  acc_cnt_ = 0;
  const OperatorConf op_conf =
      proto.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
  acc_num_ = user_op::UserOpConfWrapper(op_conf).attr<int32_t>("acc_num");
  OF_SET_MSG_HANDLER(&GradAccActor::HandlerNormal);
}

void GradAccActor::Act() {
  if (acc_cnt_ == 0) {
    Regst* out_regst = GetNaiveCurWriteable("out");
    Regst* in_regst = GetNaiveCurReadable("in");
    const Blob* in_blob = in_regst->GetMutSoleBlob();
    Blob* out_blob = out_regst->GetMutSoleBlob();
    const size_t size = in_blob->ByteSizeOfBlobBody();
    CHECK_EQ(out_blob->ByteSizeOfBlobBody(), size);
    AutoMemcpy(actor_ctx()->stream_ctx()->stream(), out_blob->ForceMutDptr(), in_blob->dptr(), size,
               out_blob->mem_case(), in_blob->mem_case());
  } else {
    AsyncLaunchKernel();
  }
  acc_cnt_ += 1;
  if (acc_cnt_ >= acc_num_) { acc_cnt_ = 0; }
}

REGISTER_ACTOR(TaskType::kGradAcc, GradAccActor);

}  // namespace oneflow
