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
#ifndef ONEFLOW_CORE_OPERATOR_FOREIGN_INPUT_OP_H_
#define ONEFLOW_CORE_OPERATOR_FOREIGN_INPUT_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ForeignInputOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignInputOp);
  ForeignInputOp() : Operator() {}
  ~ForeignInputOp() = default;

  void InitFromOpConf() override;
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;
  LogicalNode* NewProperLogicalNode() const override { return new ForeignInputLogicalNode; }

 private:
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_FOREIGN_INPUT_OP_H_
