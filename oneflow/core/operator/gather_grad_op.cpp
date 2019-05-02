#include "oneflow/core/operator/gather_grad_op.h"
#include "oneflow/core/operator/gather_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void GatherGradOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_grad_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("out_diff", false);
  EnrollOutputBn("in_diff", false);
}

const PbMessage& GatherGradOp::GetCustomizedConf() const { return op_conf().gather_grad_conf(); }

void GatherGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const GatherGradOpConf& conf = op_conf().gather_grad_conf();
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  const BlobDesc* out_diff = GetBlobDesc4BnInOp("out_diff");
  std::vector<int64_t> in_diff_dim_vec;
  in_diff_dim_vec.insert(in_diff_dim_vec.end(), out_diff->shape().dim_vec().cbegin(),
                         out_diff->shape().dim_vec().cbegin() + conf.axis());
  in_diff_dim_vec.push_back(conf.gather_dim_size());
  in_diff_dim_vec.insert(
      in_diff_dim_vec.end(),
      out_diff->shape().dim_vec().cbegin() + conf.axis() + indices->shape().NumAxes(),
      out_diff->shape().dim_vec().end());
  BlobDesc* in_diff = GetBlobDesc4BnInOp("in_diff");
  in_diff->set_data_type(out_diff->data_type());
  in_diff->mut_shape() = Shape(in_diff_dim_vec);
}

void GatherGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  // TODO: complete other signatures
}

void GatherGradOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("in_diff") = false;
}

REGISTER_OP(OperatorConf::kGatherGradConf, GatherGradOp);

}  // namespace oneflow
