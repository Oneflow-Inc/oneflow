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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {

class DistributeAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeAddOp);
  DistributeAddOp() = default;
  ~DistributeAddOp() = default;

  void InitFromOpConf() override;

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;
  LogicalNode* NewProperLogicalNode() const override { return new DistributeConcatLogicalNode; }

 private:
  Maybe<void> InferBlobParallelDesc() override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
};

void DistributeAddOp::InitFromOpConf() {
  CHECK(op_conf().has_distribute_add_conf());

  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> DistributeAddOp::InferBlobParallelDesc() {
  HashMap<std::string, std::shared_ptr<const ParallelDesc>> bn2parallel_desc;
  const std::shared_ptr<const ParallelDesc> op_parallel_desc = JUST(GetOpParallelDesc());
  FOR_RANGE(int, i, 0, input_bns().size()) {
    bn2parallel_desc[input_bns().Get(i)] =
        std::make_shared<const ParallelDesc>(op_parallel_desc->GetParallelIdOnlyParallelConf(i));
  }
  bn2parallel_desc["out"] = op_parallel_desc;
  FillBlobParallelDesc([&](const std::string& bn) -> Maybe<const ParallelDesc> {
    auto it = bn2parallel_desc.find(bn);
    CHECK_OR_RETURN(it != bn2parallel_desc.end());
    return it->second;
  });
  return Maybe<void>::Ok();
}

Maybe<void> DistributeAddOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  const BlobDesc* in_0 = BlobDesc4BnInOp(input_bns().Get(0));
  FOR_RANGE(int, i, 1, output_bns().size()) {
    const BlobDesc* in_i = BlobDesc4BnInOp(input_bns().Get(i));
    CHECK_OR_RETURN(*in_i == *in_0);
  }
  *BlobDesc4BnInOp("out") = *in_0;
  return Maybe<void>::Ok();
}

Maybe<void> DistributeAddOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const BlobDesc* first_blob_desc = nullptr;
  FOR_RANGE(int, i, 0, input_bns().size()) {
    first_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    if (first_blob_desc != nullptr) { break; }
  }
  CHECK_NOTNULL(first_blob_desc);
  *GetBlobDesc4BnInOp("out") = *first_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> DistributeAddOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), input_bns().size());
  const auto& first_in_hint = *JUST(SbpInferHint4Ibn(input_bns().Get(0)));
  FOR_RANGE(int, i, 0, input_bns().size()) {
    const auto& in_sbp_infer_hint = *JUST(SbpInferHint4Ibn(input_bns().Get(i)));
    CHECK_EQ_OR_RETURN(1, in_sbp_infer_hint.parallel_desc().parallel_num());
    CHECK_EQ_OR_RETURN(first_in_hint.logical_blob_desc().shape(),
                       in_sbp_infer_hint.logical_blob_desc().shape());
  }
  auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
  for (const auto& ibn : input_bns()) { (*bn2sbp)[ibn].mutable_partial_sum_parallel(); }
  (*bn2sbp)["out"].mutable_partial_sum_parallel();
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDistributeAddConf, DistributeAddOp);
REGISTER_DISABLE_INPUT_BOXING_GROUP(OperatorConf::kDistributeAddConf);

}  // namespace oneflow
