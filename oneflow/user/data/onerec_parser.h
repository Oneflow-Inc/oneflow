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
#ifndef ONEFLOW_CUSTOMIZED_DATA_ONEREC_PARSER_H_
#define ONEFLOW_CUSTOMIZED_DATA_ONEREC_PARSER_H_

#include "oneflow/user/data/parser.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/user/kernels/example_generated.h"

namespace oneflow {
namespace data {

class OneRecParser final : public Parser<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  OneRecParser() = default;
  ~OneRecParser() = default;

  void Parse(std::shared_ptr<LoadTargetPtrList> batch_data,
             user_op::KernelComputeContext* ctx) override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const bool verify_example = ctx->Attr<bool>("verify_example");
    FOR_RANGE(int32_t, i, 0, batch_data->size()) {
      TensorBuffer* tensor = batch_data->at(i).get();
      if (verify_example) {
        flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(tensor->data()),
                                       static_cast<size_t>(tensor->elem_cnt()));
        CHECK(onerec::example::VerifyExampleBuffer(verifier));
      }
      TensorBuffer* out = out_tensor->mut_dptr<TensorBuffer>() + i;
      out->Swap(tensor);
    }
  }
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_ONEREC_PARSER_H_
