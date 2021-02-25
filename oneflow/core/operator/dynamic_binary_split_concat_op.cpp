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
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

class DynamicBinarySplitOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicBinarySplitOp);
  DynamicBinarySplitOp() = default;
  ~DynamicBinarySplitOp() = default;

  void InitFromOpConf() override;

 private:
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void DynamicBinarySplitOp::InitFromOpConf() {
  CHECK(op_conf().has_dynamic_binary_split_conf());
  EnrollInputBn("in");
  EnrollRepeatedOutputBnWithSetter("out", [](OutputBlobModifier* ob_modifier) {
    ob_modifier->set_header_infered_before_compute(false);
  });
}

Maybe<void> DynamicBinarySplitOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const auto& in_blob_desc = *GetBlobDesc4BnInOp("in");
  CHECK_OR_RETURN(in_blob_desc.is_dynamic());
  CHECK_GE_OR_RETURN(output_bns().size(), 2);
  int32_t base_shift = op_conf().dynamic_binary_split_conf().base_shift();
  std::vector<int64_t> out_sizes(output_bns().size(), -1);
  int64_t base_size = static_cast<int64_t>(1) << base_shift;
  int64_t current_size = base_size;
  int64_t total_size = 0;
  FOR_RANGE(int, i, 0, output_bns().size()) {
    out_sizes.at(i) = current_size;
    total_size += current_size;
    if (i > 0) { current_size = current_size << 1; }
  }
  CHECK_EQ_OR_RETURN(total_size, current_size);
  int64_t in_blob_size = RtBlobDesc(in_blob_desc).AlignedTotalByteSize();
  CHECK_LE_OR_RETURN(in_blob_size, total_size);
  FOR_RANGE(int, i, 0, output_bns().size()) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    CHECK_OR_RETURN(blob_desc != nullptr);
    *blob_desc = in_blob_desc;
    // out blob shape sort from large to small like 32,16,8,8
    blob_desc->mut_shape() = Shape({out_sizes.at(output_bns().size() - 1 - i)});
    blob_desc->set_data_type(DataType::kChar);
  }
  return Maybe<void>::Ok();
}

Maybe<void> DynamicBinarySplitOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in")).shape().NumAxes();
  for (int32_t i = 0; i < num_axes; ++i) {
    SbpSignatureBuilder()
        .Split(input_bns(), i)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

class DynamicBinaryConcatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicBinaryConcatOp);
  DynamicBinaryConcatOp() = default;
  ~DynamicBinaryConcatOp() = default;

  void InitFromOpConf() override;

 private:
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
};

void DynamicBinaryConcatOp::InitFromOpConf() {
  CHECK(op_conf().has_dynamic_binary_concat_conf());
  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> DynamicBinaryConcatOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const auto& conf = op_conf().dynamic_binary_concat_conf();
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *GetBlobDesc4BnInOp(input_bns().Get(0));
  out_blob_desc->set_data_type(conf.out_data_type());
  out_blob_desc->mut_shape() = Shape(conf.out_shape());
  // check valid
  CHECK_GE_OR_RETURN(input_bns().size(), 2);
  CHECK_OR_RETURN(out_blob_desc->is_dynamic());
  int64_t input_total_size = 0;
  for (const auto& ibn : input_bns()) {
    const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(ibn);
    CHECK_EQ_OR_RETURN(in_blob_desc->data_type(), DataType::kChar);
    input_total_size += RtBlobDesc(*in_blob_desc).ByteSizeOfBlobBody();
  }
  CHECK_GE_OR_RETURN(input_total_size, RtBlobDesc(*out_blob_desc).AlignedTotalByteSize());
  return Maybe<void>::Ok();
}

Maybe<void> DynamicBinaryConcatOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  // DO NOTHING
  return Maybe<void>::Ok();
}

Maybe<void> DynamicBinaryConcatOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())[ibn] =
        JUST(SbpInferHint4Ibn(ibn))->sbp_parallel();
  }
  (*sbp_signature->mutable_bn_in_op2sbp_parallel())["out"] =
      op_conf().dynamic_binary_concat_conf().out_sbp();
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDynamicBinarySplitConf, DynamicBinarySplitOp);
REGISTER_OP(OperatorConf::kDynamicBinaryConcatConf, DynamicBinaryConcatOp);

}  // namespace oneflow
