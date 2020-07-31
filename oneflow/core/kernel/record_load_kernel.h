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
#ifndef ONEFLOW_CORE_KERNEL_RECORD_LOAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RECORD_LOAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/record/ofrecord_reader.h"

namespace oneflow {

struct RecordLoadStatus {
  bool is_eof;
  int64_t record_num;
};

class RecordLoadKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordLoadKernel);
  RecordLoadKernel() = default;
  ~RecordLoadKernel() override = default;

  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  void VirtualKernelInit() override;

  std::unique_ptr<PersistentInStream> in_stream_;
  std::unique_ptr<OFRecordReader> record_reader_;
  int64_t piece_size_in_one_loader_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RECORD_LOAD_KERNEL_H_
