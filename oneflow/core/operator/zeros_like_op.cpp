#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ZerosLikeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ZerosLikeOp);
  ZerosLikeOp() = default;
  ~ZerosLikeOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_zeros_like_conf());
    EnrollOutputBn("out");
    EnrollInputBn("like", false)->set_use_header_only(true);
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().zeros_like_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* like_blob_desc = GetBlobDesc4BnInOp("like");
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    GetBlobDesc4BnInOp("out")->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
    if (op_conf().zeros_like_conf().data_type()) {
      out_blob_desc->set_data_type(op_conf().zeros_like_conf().data_type());
    } else {
      out_blob_desc->set_data_type(like_blob_desc->data_type());
    }
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .MakeSplitSignatureListBuilder(
            JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kZerosLikeConf, ZerosLikeOp);

}  // namespace oneflow
