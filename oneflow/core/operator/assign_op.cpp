#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AssignOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AssignOp);
  AssignOp() = default;
  ~AssignOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void AssignOp::InitFromOpConf() {
  CHECK(op_conf().has_assign_conf());
  EnrollInputBn("ref")->set_is_mutable(true);
  EnrollInputBn("value");
}

const PbMessage& AssignOp::GetCustomizedConf() const { return op_conf().assign_conf(); }

std::string DebugString(const BlobDesc& blob_desc) {
  BlobDescProto blob_desc_proto;
  blob_desc.ToProto(&blob_desc_proto);
  return blob_desc_proto.DebugString();
}

Maybe<void> AssignOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("ref") == *GetBlobDesc4BnInOp("value"))
      << "\nref_blob_desc: " << DebugString(*GetBlobDesc4BnInOp("ref"))
      << "\nvalue_blob_desc: " << DebugString(*GetBlobDesc4BnInOp("value"));
  return Maybe<void>::Ok();
}

Maybe<void> AssignOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .MakeSplitSignatureListBuilder(
          JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kAssignConf, AssignOp);

}  // namespace oneflow
