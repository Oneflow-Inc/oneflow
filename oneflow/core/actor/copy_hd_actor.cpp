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
#include "oneflow/core/actor/copy_hd_actor.h"

namespace oneflow {

#ifdef WITH_CUDA

void CopyHdActor::VirtualActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&CopyHdActor::HandlerNormal);
}

void CopyHdActor::Act() { AsyncLaunchKernel(GenDefaultKernelCtx()); }

void CopyHdActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  Regst* in_regst = GetNaiveCurReadable("copy_in");
  HandleProducedNaiveDataRegstToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    return true;
  });
}

REGISTER_ACTOR(TaskType::kCopyHd, CopyHdActor);

#endif

}  // namespace oneflow
