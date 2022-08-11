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
#ifndef ONEFLOW_USER_DATA_OFRECORD_IMAGE_CLASSIFICATION_PARSER_H_
#define ONEFLOW_USER_DATA_OFRECORD_IMAGE_CLASSIFICATION_PARSER_H_

#include "oneflow/user/data/parser.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/user/data/ofrecord_image_classification_dataset.h"

namespace oneflow {

namespace data {

class OFRecordImageClassificationParser final : public Parser<ImageClassificationDataInstance> {
 public:
  using Base = Parser<ImageClassificationDataInstance>;
  using SampleType = typename Base::SampleType;
  using BatchType = typename Base::BatchType;

  OFRecordImageClassificationParser() = default;
  ~OFRecordImageClassificationParser() override = default;

  void Parse(BatchType& batch_data, user_op::KernelComputeContext* ctx) override {
    const int64_t batch_size = batch_data.size();
    user_op::Tensor* image_tensor = ctx->Tensor4ArgNameAndIndex("image", 0);
    CHECK_EQ(image_tensor->shape_view().NumAxes(), 1);
    CHECK_EQ(image_tensor->shape_view().At(0), batch_size);
    auto* image_buffers = image_tensor->mut_dptr<TensorBuffer>();
    user_op::Tensor* label_tensor = ctx->Tensor4ArgNameAndIndex("label", 0);
    CHECK_EQ(label_tensor->shape_view().NumAxes(), 1);
    CHECK_EQ(label_tensor->shape_view().At(0), batch_size);
    auto* label_buffers = label_tensor->mut_dptr<TensorBuffer>();
    for (size_t i = 0; i < batch_data.size(); ++i) {
      auto& instance = batch_data[i];
      image_buffers[i].Swap(instance.image);
      label_buffers[i].Swap(instance.label);
    }
  }
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_OFRECORD_IMAGE_CLASSIFICATION_PARSER_H_
