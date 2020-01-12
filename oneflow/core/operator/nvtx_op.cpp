#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class NvtxRangeStartOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NvtxRangeStartOp);
  NvtxRangeStartOp() = default;
  ~NvtxRangeStartOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_nvtx_range_start_conf());
    int32_t in_size = op_conf().nvtx_range_start_conf().in_size();
    CHECK_GT(in_size, 0);
    EnrollRepeatedInputBn("in", in_size);
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().nvtx_range_start_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    const auto bns = StdVec2PbRpf<std::string>({"in"});
    SbpSignatureBuilder().PartialSum(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
    const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
    SbpSignatureBuilder().Split(bns, 0).MakeSplitSignatureListBuilder(num_axes).Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

class NvtxRangeEndOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NvtxRangeEndOp);
  NvtxRangeEndOp() = default;
  ~NvtxRangeEndOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_nvtx_range_end_conf());
    int32_t in_size = op_conf().nvtx_range_end_conf().in_size();
    CHECK_GT(in_size, 0);
    EnrollRepeatedInputBn("in", in_size);
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().nvtx_range_end_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    const auto bns = StdVec2PbRpf<std::string>({"in"});
    SbpSignatureBuilder().PartialSum(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
    const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
    SbpSignatureBuilder().Split(bns, 0).MakeSplitSignatureListBuilder(num_axes).Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kNvtxRangeStartConf, NvtxRangeStartOp);
REGISTER_OP(OperatorConf::kNvtxRangeEndConf, NvtxRangeEndOp);

}  // namespace oneflow
