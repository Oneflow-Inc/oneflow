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
#ifndef ONEFLOW_USER_DATA_OFRECORD_PARSER_H_
#define ONEFLOW_USER_DATA_OFRECORD_PARSER_H_

#include "oneflow/user/data/parser.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {
namespace data {

class OFRecordParser final : public Parser<TensorBuffer> {
 public:
  using Base = Parser<TensorBuffer>;
  using SampleType = typename Base::SampleType;
  using BatchType = typename Base::BatchType;

  OFRecordParser() = default;
  ~OFRecordParser() = default;

  void Parse(BatchType& batch_data, user_op::KernelComputeContext* ctx) override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    OFRecord* dptr = out_tensor->mut_dptr<OFRecord>();
    MultiThreadLoop(batch_data.size(), [&](size_t i) {
      auto& sample = batch_data[i];
      CHECK(dptr[i].ParseFromArray(sample.data(), sample.nbytes()));
    });
    if (batch_data.size() != out_tensor->shape_view().elem_cnt()) {
      CHECK_EQ(out_tensor->mut_shape_view().NumAxes(), 1);
      out_tensor->mut_shape_view().Set(0, batch_data.size());
    }
  }
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_OFRECORD_PARSER_H_
