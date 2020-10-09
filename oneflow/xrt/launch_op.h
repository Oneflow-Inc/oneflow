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
#ifndef ONEFLOW_XRT_LAUNCH_OP_H_
#define ONEFLOW_XRT_LAUNCH_OP_H_

#include <string>

#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class XrtLaunchOp : public Operator {
 public:
  void InitFromOpConf() override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

  LogicalNode* NewProperLogicalNode() const override {
    const auto& launch_conf = op_conf().xrt_launch_conf();
    if (launch_conf.model_update()) { return new OptimizerLogicalNode; }
    return new NormalForwardLogicalNode;
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;

  typedef std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4IbnFunc;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      SbpInferHint4IbnFunc SbpInferHint4Ibn, const ParallelDesc& parallel_desc) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_XRT_LAUNCH_OP_H_
