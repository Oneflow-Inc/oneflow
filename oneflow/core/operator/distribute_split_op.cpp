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

class DistributeSplitOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeSplitOp);
  DistributeSplitOp() = default;
  ~DistributeSplitOp() = default;

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

Maybe<void> DistributeSplitOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  const auto& in_blob_desc = *BlobDesc4BnInOp("in");
  CHECK_EQ(parallel_desc.parallel_num(), output_bns().size());
  const auto& conf = op_conf().distribute_split_conf();
  const int32_t split_axis = FixAxis(conf.axis(), in_blob_desc.shape().NumAxes());
  BalancedSplitter bs(in_blob_desc.shape().At(split_axis), parallel_desc.parallel_num());
  FOR_RANGE(int, i, 0, parallel_desc.parallel_num()) {
    BlobDesc* out_blob_desc = BlobDesc4BnInOp(output_bns().Get(i));
    *out_blob_desc = in_blob_desc;
    out_blob_desc->mut_shape().Set(split_axis, bs.At(i).size());
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeSplitOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
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

Maybe<void> DistributeSplitOp::InferBlobParallelDesc() {
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
