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
#include "oneflow/core/actor/accumulate_compute_actor.h"

namespace oneflow {

void AccumulateCompActor::Init(const TaskProto& task_proto, int32_t max_acc_cnt) {
  using namespace std::placeholders;
  if (GetDeviceType() == DeviceType::kCPU) {
    cpy_func_ = std::bind(Memcpy<DeviceType::kCPU>, _1, _2, _3, _4);
  } else {
#ifdef WITH_CUDA
    cpy_func_ = std::bind(Memcpy<DeviceType::kGPU>, _1, _2, _3, _4);
#else
    UNIMPLEMENTED();
#endif
  }
  OF_SET_MSG_HANDLER(&AccumulateCompActor::HandlerNormal);
  acc_cnt_ = 0;
  max_acc_cnt_ = max_acc_cnt;
  next_piece_id_ = 0;
}

int64_t AccumulateCompActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  return regst_desc_id == Name2SoleRegstDescId("acc") ? max_acc_cnt_ : 1;
}

void AccumulateCompActor::Act() {
  Regst* out_regst = GetNaiveCurWriteable("acc");
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  if (acc_cnt_ == 0) {
    Blob* out_blob = out_regst->packed_blob();
    if (GetDeviceType() == DeviceType::kCPU) {
      Memset<DeviceType::kCPU>(kernel_ctx.device_ctx, out_blob->mut_dptr(), 0,
                               out_blob->ByteSizeOfBlobBody());
    } else if (GetDeviceType() == DeviceType::kGPU) {
#ifdef WITH_CUDA
      Memset<DeviceType::kGPU>(kernel_ctx.device_ctx, out_blob->mut_dptr(), 0,
                               out_blob->ByteSizeOfBlobBody());
#else
      UNIMPLEMENTED();
#endif
    } else {
      UNIMPLEMENTED();
    }
  }
  AsyncLaunchKernel(kernel_ctx);
  acc_cnt_ += 1;
}

void AccumulateCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (acc_cnt_ == max_acc_cnt_) {
    HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
      regst->set_piece_id(next_piece_id_);
      return true;
    });
    acc_cnt_ = 0;
    next_piece_id_ += 1;
  }
}

}  // namespace oneflow
