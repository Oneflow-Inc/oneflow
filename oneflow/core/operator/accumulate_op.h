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
#ifndef ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AccumulateOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccumulateOp);
  AccumulateOp() = default;
  ~AccumulateOp() = default;

  void InitFromOpConf() override;

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    return Maybe<void>::Ok();
  }
  Maybe<void> InferOutputBlobTimeShape(
      std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
      const ParallelContext* parallel_ctx, Shape* time_shape) const override {
    TODO();
    return Maybe<void>::Ok();
  }

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId lbi4obn(const std::string& output_bn) const override { return GenPackedLbi(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_
