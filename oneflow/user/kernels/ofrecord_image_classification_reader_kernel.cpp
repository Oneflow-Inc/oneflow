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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/data/ofrecord_image_classification_data_reader.h"

namespace oneflow {

namespace {

class OFRecordImageClassificationReaderKernelState final : public user_op::OpKernelState {
 public:
  explicit OFRecordImageClassificationReaderKernelState(user_op::KernelInitContext* ctx)
      : reader_(ctx) {}
  ~OFRecordImageClassificationReaderKernelState() override = default;

  void Read(user_op::KernelComputeContext* ctx) { reader_.Read(ctx); }

 private:
  data::OFRecordImageClassificationDataReader reader_;
};

}  // namespace

class OFRecordImageClassificationReaderKernel final : public user_op::OpKernel {
 public:
  OFRecordImageClassificationReaderKernel() = default;
  ~OFRecordImageClassificationReaderKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<OFRecordImageClassificationReaderKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* reader = dynamic_cast<OFRecordImageClassificationReaderKernelState*>(state);
    CHECK_NOTNULL(reader);
    reader->Read(ctx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("ofrecord_image_classification_reader")
    .SetCreateFn<OFRecordImageClassificationReaderKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)
                     & (user_op::HobDataType("image", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("label", 0) == DataType::kTensorBuffer));

}  // namespace oneflow
