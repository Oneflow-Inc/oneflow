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

class DistributeConcatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeConcatOp);
  DistributeConcatOp() = default;
  ~DistributeConcatOp() = default;

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

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;

  int32_t FixAxis(const int32_t axis, const int64_t num_axes) const;
};

void DistributeConcatOp::InitFromOpConf() {
  CHECK(op_conf().has_distribute_concat_conf());

  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> DistributeConcatOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  if (parallel_ctx->parallel_num() > 1) {
    const auto* in_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(parallel_ctx->parallel_id()));
    *GetBlobDesc4BnInOp("out") = *in_blob_desc;
    return Maybe<void>::Ok();
  }
  const auto& conf = op_conf().distribute_concat_conf();
  const BlobDesc* first_blob_desc = nullptr;
  int first_blob_desc_idx = -1;
  FOR_RANGE(int, i, 0, input_bns().size()) {
    first_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    if (first_blob_desc != nullptr) {
      first_blob_desc_idx = i;
      break;
    }
  }
  CHECK_NOTNULL(first_blob_desc);
  DimVector out_dim_vec = first_blob_desc->shape().dim_vec();
  int32_t concat_axis = FixAxis(conf.axis(), out_dim_vec.size());
  for (size_t i = 0; i < input_bns().size(); ++i) {
    const BlobDesc* in_i_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    if (in_i_blob_desc == nullptr) { continue; }
    if (first_blob_desc_idx == i) { continue; }
    for (int64_t j = 0; j < in_i_blob_desc->shape().NumAxes(); ++j) {
      if (j == concat_axis) {
        out_dim_vec[j] += in_i_blob_desc->shape().At(j);
      } else {
        CHECK_EQ_OR_RETURN(out_dim_vec[j], in_i_blob_desc->shape().At(j));
      }
    }
    CHECK_EQ_OR_RETURN(in_i_blob_desc->data_type(), first_blob_desc->data_type());
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *first_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> DistributeConcatOp::InferParallelSignature() {
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

Maybe<void> DistributeConcatOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  FOR_RANGE(int32_t, i, 0, input_bns().size()) {
    CHECK_OR_RETURN(*BatchAxis4BnInOp(input_bns().Get(i)) == *BatchAxis4BnInOp(input_bns().Get(0)));
  }
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp(input_bns().Get(0));
  return Maybe<void>::Ok();
}

Maybe<void> DistributeConcatOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), input_bns().size());
  auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> Maybe<const BlobDesc&> {
    const SbpInferHint* sbp_infer_hint = JUST(SbpInferHint4Ibn(ibn));
    return Maybe<const BlobDesc&>(sbp_infer_hint->logical_blob_desc());
  };
  {
    // check parallel_num and dimention
    const auto& conf = op_conf().distribute_concat_conf();
    const int64_t num_axes = JUST(LogicalBlobDesc4Ibn(input_bns().Get(0))).shape().NumAxes();
    const int32_t axis = FixAxis(conf.axis(), num_axes);
    int64_t dim = 0;
    FOR_RANGE(int, i, 0, input_bns().size()) {
      const auto& in_parallel_desc = JUST(SbpInferHint4Ibn(input_bns().Get(i)))->parallel_desc();
      CHECK_EQ_OR_RETURN(1, in_parallel_desc.parallel_num());
      dim += JUST(LogicalBlobDesc4Ibn(input_bns().Get(i))).shape().At(axis);
    }
    BalancedSplitter bs(dim, parallel_desc.parallel_num());
    FOR_RANGE(int, i, 0, input_bns().size()) {
      CHECK_EQ_OR_RETURN(JUST(LogicalBlobDesc4Ibn(input_bns().Get(i))).shape().At(axis),
                         bs.At(i).size());
    }
  }
  SbpSignatureList sbp_sig_list;
  GetSbpSignatures(LogicalBlobDesc4Ibn, &sbp_sig_list);
  *sbp_signature = sbp_sig_list.sbp_signature().Get(0);
  return Maybe<void>::Ok();
}

Maybe<void> DistributeConcatOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto& conf = op_conf().distribute_concat_conf();
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn(input_bns().Get(0))).shape().NumAxes();
  const int32_t axis = FixAxis(conf.axis(), num_axes);
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), axis)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

int32_t DistributeConcatOp::FixAxis(const int32_t axis, const int64_t num_axes) const {
  int32_t ret = axis;
  if (axis < 0) { ret += num_axes; }
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return ret;
}

REGISTER_OP(OperatorConf::kDistributeConcatConf, DistributeConcatOp);
REGISTER_DISABLE_INPUT_BOXING_GROUP(OperatorConf::kDistributeConcatConf);

}  // namespace oneflow
