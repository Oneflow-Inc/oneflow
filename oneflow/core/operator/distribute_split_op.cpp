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

class DistributeSplitOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeSplitOp);
  DistributeSplitOp() = default;
  ~DistributeSplitOp() = default;

  void InitFromOpConf() override;

  LogicalNode* NewProperLogicalNode() const override { return new DistributeSplitLogicalNode; }

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
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override;
  Maybe<void> InferOutParallelDesc(
      std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn,
      std::function<const BlobDesc*(const std::string&)> LogicalBlobDesc4Ibn, const ParallelDesc&,
      const SbpSignature*) const override;

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;

  int32_t FixAxis(const int32_t axis, const int64_t num_axes) const;
};

void DistributeSplitOp::InitFromOpConf() {
  CHECK(op_conf().has_distribute_split_conf());
  EnrollInputBn("in");
  EnrollRepeatedOutputBnWithSetter("out", [&](OutputBlobModifier* ob_modifier) {
    ob_modifier->set_header_infered_before_compute(false);
    ob_modifier->set_is_mutable(op_conf().distribute_split_conf().is_variable_ref());
  });
}

Maybe<void> DistributeSplitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& in_blob_desc = *GetBlobDesc4BnInOp("in");
  if (parallel_ctx->parallel_num() > 1) {
    CHECK_EQ(parallel_ctx->parallel_num(), output_bns().size());
    auto* out_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(parallel_ctx->parallel_id()));
    *out_blob_desc = in_blob_desc;
    return Maybe<void>::Ok();
  }
  const auto& conf = op_conf().distribute_split_conf();
  int32_t split_axis = FixAxis(conf.axis(), in_blob_desc.shape().NumAxes());
  std::vector<BlobDesc*> out_blob_descs;
  FOR_RANGE(int, i, 0, output_bns().size()) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    if (blob_desc != nullptr) { out_blob_descs.push_back(blob_desc); }
  }
  BalancedSplitter bs(in_blob_desc.shape().At(split_axis), out_blob_descs.size());
  FOR_RANGE(int, i, 0, out_blob_descs.size()) {
    *out_blob_descs.at(i) = in_blob_desc;
    out_blob_descs.at(i)->mut_shape().Set(split_axis, bs.At(i).size());
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeSplitOp::InferOutParallelDesc(
    std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn,
    std::function<const BlobDesc*(const std::string&)> LogicalBlobDesc4Ibn,
    const ParallelDesc& op_parallel_desc, const SbpSignature*) const {
  FOR_RANGE(int, i, 0, output_bns().size()) {
    const auto& obn = output_bns().Get(i);
    if (op_parallel_desc.parallel_num() > 1) {
      CHECK_EQ(op_parallel_desc.parallel_num(), output_bns().size());
      *ParallelDesc4Obn(obn) = ParallelDesc(op_parallel_desc.GetParallelIdOnlyParallelConf(i));
    } else {
      *ParallelDesc4Obn(obn) = op_parallel_desc;
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeSplitOp::InferParallelSignature() {
  const auto& scope_storage = *Global<vm::SymbolStorage<Scope>>::Get();
  const auto& scope = JUST(scope_storage.MaybeGet(op_conf().scope_symbol_id()));
  int64_t op_parallel_desc_symbol_id = JUST(scope.GetParallelDescSymbolId(op_conf()));
  mut_parallel_signature()->set_op_parallel_desc_symbol_id(op_parallel_desc_symbol_id);
  auto* map = mut_parallel_signature()->mutable_bn_in_op2parallel_desc_symbol_id();
  (*map)["in"] = op_parallel_desc_symbol_id;
  const auto& op_parallel_desc = JUST(scope.GetParallelDesc(op_conf()));
  CHECK_EQ(op_parallel_desc.parallel_num(), output_bns().size());
  FOR_RANGE(int, i, 0, output_bns().size()) {
    const auto& out_parallel_conf = op_parallel_desc.GetParallelIdOnlyParallelConf(i);
    const std::shared_ptr<cfg::ParallelConf>& cfg_out_parallel_conf =
        std::make_shared<cfg::ParallelConf>(out_parallel_conf);
    (*map)[output_bns().Get(i)] =
        Global<ForeignCallback>::Get()->MakeParallelDescSymbol(cfg_out_parallel_conf);
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeSplitOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  FOR_RANGE(int32_t, i, 0, output_bns().size()) {
    *BatchAxis4BnInOp(output_bns().Get(i)) = *BatchAxis4BnInOp("in");
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeSplitOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), output_bns().size());
  auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> Maybe<const BlobDesc&> {
    const SbpInferHint* sbp_infer_hint = JUST(SbpInferHint4Ibn(ibn));
    return Maybe<const BlobDesc&>(sbp_infer_hint->logical_blob_desc());
  };
  SbpSignatureList sbp_sig_list;
  GetSbpSignatures(LogicalBlobDesc4Ibn, &sbp_sig_list);
  *sbp_signature = sbp_sig_list.sbp_signature().Get(0);
  return Maybe<void>::Ok();
}

Maybe<void> DistributeSplitOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto& conf = op_conf().distribute_split_conf();
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in")).shape().NumAxes();
  const int32_t axis = FixAxis(conf.axis(), num_axes);
  SbpSignatureBuilder()
      .Split(input_bns(), axis)
      .Broadcast(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

int32_t DistributeSplitOp::FixAxis(const int32_t axis, const int64_t num_axes) const {
  int32_t ret = axis;
  if (axis < 0) { ret += num_axes; }
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return ret;
}

REGISTER_OP(OperatorConf::kDistributeSplitConf, DistributeSplitOp);

}  // namespace oneflow
