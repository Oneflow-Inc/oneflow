#include "oneflow/core/operator/case_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void CaseOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollRepeatedOutputBn("out", false);
}

Maybe<void> CaseOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in->shape().elem_cnt(), 1);
  const DataType data_type = in->data_type();
  CHECK_OR_RETURN(IsIntegralDataType(data_type));
  for (const std::string& obn : output_bns()) {
    BlobDesc* out = GetBlobDesc4BnInOp(obn);
    out->mut_shape() = Shape({1});
    out->set_data_type(data_type);
  }
  return Maybe<void>::Ok();
}

const PbMessage& CaseOp::GetCustomizedConf() const { return op_conf().case_conf(); }

Maybe<void> CaseOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const std::string& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp("in"); }
  return Maybe<void>::Ok();
}

void CaseOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {}

LogicalNode* CaseOp::NewProperLogicalNode() const { return new CaseLogicalNode(); }

REGISTER_CPU_OP(OperatorConf::kCaseConf, CaseOp);

}  // namespace oneflow
