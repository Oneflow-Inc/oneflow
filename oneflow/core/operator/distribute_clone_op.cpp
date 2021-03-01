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

class DistributeCloneOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeCloneOp);
  DistributeCloneOp() = default;
  ~DistributeCloneOp() = default;

  void InitFromOpConf() override;

  LogicalNode* NewProperLogicalNode() const override { return new DistributeSplitLogicalNode; }

 private:
  Maybe<void> InferBlobParallelDesc() override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;
};

void DistributeCloneOp::InitFromOpConf() {
  CHECK(op_conf().has_distribute_clone_conf());

  EnrollInputBn("in");
  EnrollRepeatedOutputBnWithSetter("out", [&](OutputBlobModifier* ob_modifier) {
    ob_modifier->set_is_mutable(op_conf().distribute_clone_conf().is_variable_ref());
  });
}

Maybe<void> DistributeCloneOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  const auto& in_blob_desc = *BlobDesc4BnInOp("in");
  FOR_RANGE(int, i, 0, output_bns().size()) {
    BlobDesc* blob_desc = BlobDesc4BnInOp(output_bns().Get(i));
    *blob_desc = in_blob_desc;
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeCloneOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const auto& in_blob_desc = *GetBlobDesc4BnInOp("in");
  if (parallel_ctx->parallel_num() > 1) {
    CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), output_bns().size());
    auto* out_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(parallel_ctx->parallel_id()));
    *out_blob_desc = in_blob_desc;
    return Maybe<void>::Ok();
  }
  FOR_RANGE(int, i, 0, output_bns().size()) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    if (blob_desc != nullptr) { *blob_desc = in_blob_desc; }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeCloneOp::InferBlobParallelDesc() {
  HashMap<std::string, std::shared_ptr<const ParallelDesc>> bn2parallel_desc;
  const std::shared_ptr<const ParallelDesc> op_parallel_desc = JUST(GetOpParallelDesc());
  bn2parallel_desc["in"] = op_parallel_desc;
  FOR_RANGE(int, i, 0, output_bns().size()) {
    bn2parallel_desc[output_bns().Get(i)] =
        std::make_shared<const ParallelDesc>(op_parallel_desc->GetParallelIdOnlyParallelConf(i));
  }
  FillBlobParallelDesc([&](const std::string& bn) -> Maybe<const ParallelDesc> {
    auto it = bn2parallel_desc.find(bn);
    CHECK_OR_RETURN(it != bn2parallel_desc.end());
    return it->second;
  });
  return Maybe<void>::Ok();
}

Maybe<void> DistributeCloneOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), output_bns().size());
  const SbpInferHint& in_hint = *JUST(SbpInferHint4Ibn("in"));
  CHECK_OR_RETURN(in_hint.parallel_desc() == parallel_desc);
  SbpSignatureBuilder().Broadcast(output_bns()).Build(sbp_signature);
  auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
  (*bn2sbp)["in"].mutable_broadcast_parallel();
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDistributeCloneConf, DistributeCloneOp);

}  // namespace oneflow
