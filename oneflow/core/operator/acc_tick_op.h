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
#ifndef ONEFLOW_CORE_OPERATOR_ACC_TICK_OP_H_
#define ONEFLOW_CORE_OPERATOR_ACC_TICK_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AccTickOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccTickOp);
  AccTickOp() = default;
  ~AccTickOp() = default;

  void InitFromOpConf() override;

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferOpTimeShape(
      const std::function<Maybe<const Shape>(const std::string&)>& GetTimeShape4BnInOp,
      std::shared_ptr<const Shape>* time_shape) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACC_TICK_OP_H_
