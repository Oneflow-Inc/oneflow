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
#ifndef ONEFLOW_USER_DATA_GPT_PARSER_H_
#define ONEFLOW_USER_DATA_GPT_PARSER_H_

#include "oneflow/user/data/parser.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

namespace data {

class GPTParser final : public Parser<TensorBuffer> {
 public:
  using LoadTargetShdPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  GPTParser() = default;
  ~GPTParser() = default;

  void Parse(std::shared_ptr<LoadTargetShdPtrVec> batch_data,
             user_op::KernelComputeContext* ctx) override {
    user_op::Tensor* sequence_tensor = ctx->Tensor4ArgNameAndIndex("sequence", 0);
    CHECK_NOTNULL(sequence_tensor);
    auto* seq_buf = sequence_tensor->mut_dptr<TensorBuffer>();
    MultiThreadLoop(batch_data->size(), [&](size_t i) { seq_buf->Swap(batch_data->at(i).get()); });
    if (batch_data->size() != sequence_tensor->shape().elem_cnt()) {
      CHECK_EQ(sequence_tensor->mut_shape()->NumAxes(), 1);
      sequence_tensor->mut_shape()->Set(0, batch_data->size());
    }
  }
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_GPT_PARSER_H_
