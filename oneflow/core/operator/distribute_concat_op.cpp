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

class DistributeConcatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeConcatOp);
  DistributeConcatOp() = default;
  ~DistributeConcatOp() = default;

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

Maybe<void> DistributeConcatOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  const auto& conf = op_conf().distribute_concat_conf();
  BlobDesc* out = BlobDesc4BnInOp("out");
  *out = *BlobDesc4BnInOp(input_bns().Get(0));
  const int32_t concat_axis = FixAxis(conf.axis(), out->shape().NumAxes());
  int64_t concat_dim_size = out->shape().At(concat_axis);
  for (size_t i = 1; i < input_bns().size(); ++i) {
    const BlobDesc* in_i = BlobDesc4BnInOp(input_bns().Get(i));
    for (int64_t j = 0; j < in_i->shape().NumAxes(); ++j) {
      if (j == concat_axis) {
        concat_dim_size += in_i->shape().At(j);
      } else {
        CHECK_EQ_OR_RETURN(out->shape().At(j), in_i->shape().At(j));
      }
    }
    CHECK_EQ_OR_RETURN(in_i->data_type(), out->data_type());
  }
  out->mut_shape().Set(concat_axis, concat_dim_size);
  out->set_is_dynamic(false);
  return Maybe<void>::Ok();
}

Maybe<void> DistributeConcatOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  if (parallel_ctx->parallel_num() > 1) {
    const auto* in_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(parallel_ctx->parallel_id()));
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    *out_blob_desc = *in_blob_desc;
    out_blob_desc->set_is_dynamic(false);
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
  out_blob_desc->set_is_dynamic(false);
  return Maybe<void>::Ok();
}

Maybe<void> DistributeConcatOp::InferBlobParallelDesc() {
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
