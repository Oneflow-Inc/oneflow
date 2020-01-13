#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

class DynamicBinarySplitOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicBinarySplitOp);
  DynamicBinarySplitOp() = default;
  ~DynamicBinarySplitOp() = default;

  void InitFromOpConf() override;

  const PbMessage& GetCustomizedConf() const override;

 private:
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void DynamicBinarySplitOp::InitFromOpConf() {
  CHECK(op_conf().has_dynamic_binary_split_conf());
  EnrollInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage& DynamicBinarySplitOp::GetCustomizedConf() const {
  return op_conf().dynamic_binary_split_conf();
}

Maybe<void> DynamicBinarySplitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& in_blob_desc = *GetBlobDesc4BnInOp("in");
  int32_t base_shift = op_conf().dynamic_binary_split_conf().base_shift();
  std::vector<int64_t> out_sizes(output_bns().size(), -1);
  int64_t base_size = static_cast<int64_t>(1) << base_shift;
  int64_t total_size = 0;
  FOR_RANGE(int, i, 0, output_bns().size()) {
    out_sizes.at(i) = base_size;
    total_size += base_size;
    if (i > 0) { base_size = base_size << 1; }
  }
  CHECK_EQ(total_size, base_size);
  int64_t in_blob_size = RtBlobDesc(in_blob_desc).AlignedTotalByteSize();
  CHECK_LE(in_blob_size, total_size);
  FOR_RANGE(int, i, 0, output_bns().size()) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    CHECK(blob_desc != nullptr);
    *blob_desc = in_blob_desc;
    blob_desc->mut_shape() = Shape({out_sizes.at(i)});
    blob_desc->set_data_type(DataType::kChar);
  }
  return Maybe<void>::Ok();
}

Maybe<void> DynamicBinarySplitOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  // out blob has NOT  batch axis
  FOR_RANGE(int32_t, i, 0, output_bns().size()) { BatchAxis4BnInOp(output_bns().Get(i)); }
  return Maybe<void>::Ok();
}

Maybe<void> DynamicBinarySplitOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
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

  const PbMessage& GetCustomizedConf() const override;

 private:
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
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

const PbMessage& DynamicBinaryConcatOp::GetCustomizedConf() const {
  return op_conf().dynamic_binary_concat_conf();
}

Maybe<void> DynamicBinaryConcatOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().dynamic_binary_concat_conf();
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *GetBlobDesc4BnInOp(input_bns().Get(0));
  out_blob_desc->set_data_type(conf.out_data_type());
  out_blob_desc->mut_shape() = Shape(conf.out_shape());
  return Maybe<void>::Ok();
}

Maybe<void> DynamicBinaryConcatOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = op_conf().dynamic_binary_concat_conf().out_batch_axis();
  return Maybe<void>::Ok();
}

Maybe<void> DynamicBinaryConcatOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  // DO NOTHING
  return Maybe<void>::Ok();
}

Maybe<void> DynamicBinaryConcatOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  *sbp_signature = op_conf().dynamic_binary_concat_conf().out_sbp_sig();
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDynamicBinarySplitConf, DynamicBinarySplitOp);
REGISTER_OP(OperatorConf::kDynamicBinaryConcatConf, DynamicBinaryConcatOp);

}  // namespace oneflow
