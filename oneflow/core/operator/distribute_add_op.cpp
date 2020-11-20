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
#include "oneflow/core/eager/eager_symbol_storage.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {

class DistributeAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeAddOp);
  DistributeAddOp() = default;
  ~DistributeAddOp() = default;

  void InitFromOpConf() override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override;
  LogicalNode* NewProperLogicalNode() const override { return new DistributeConcatLogicalNode; }

 private:
  Maybe<void> InferParallelSignature() override;
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
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

Maybe<void> DistributeAddOp::InferParallelSignature() {
  const auto& scope_storage = *Global<vm::SymbolStorage<Scope>>::Get();
  const auto& scope = JUST(scope_storage.MaybeGet(op_conf().scope_symbol_id()));
  int64_t op_parallel_desc_symbol_id = JUST(scope.GetParallelDescSymbolId(op_conf()));
  mut_parallel_signature()->set_op_parallel_desc_symbol_id(op_parallel_desc_symbol_id);
  auto* map = mut_parallel_signature()->mutable_bn_in_op2parallel_desc_symbol_id();
  (*map)["out"] = op_parallel_desc_symbol_id;
  const auto& op_parallel_desc = JUST(scope.GetParallelDesc(op_conf()));
  CHECK_EQ(op_parallel_desc.parallel_num(), input_bns().size());
  FOR_RANGE(int, i, 0, input_bns().size()) {
    const auto& in_parallel_conf = op_parallel_desc.GetParallelIdOnlyParallelConf(i);
    const std::shared_ptr<cfg::ParallelConf>& cfg_in_parallel_conf =
        std::make_shared<cfg::ParallelConf>(in_parallel_conf);
    (*map)[input_bns().Get(i)] =
        Global<ForeignCallback>::Get()->MakeParallelDescSymbol(cfg_in_parallel_conf);
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeAddOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* first_blob_desc = nullptr;
  FOR_RANGE(int, i, 0, input_bns().size()) {
    first_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    if (first_blob_desc != nullptr) { break; }
  }
  CHECK_NOTNULL(first_blob_desc);
  *GetBlobDesc4BnInOp("out") = *first_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> DistributeAddOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  FOR_RANGE(int32_t, i, 0, input_bns().size()) {
    CHECK_OR_RETURN(*BatchAxis4BnInOp(input_bns().Get(i)) == *BatchAxis4BnInOp(input_bns().Get(0)));
  }
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp(input_bns().Get(0));
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
