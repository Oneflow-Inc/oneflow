#include "oneflow/core/operator/dropout_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void DropoutOp::InitFromOpConf() {
  if (op_conf().dropout_conf().has_noise_shape()) { TODO(); }
  double dropout_rate = op_conf().dropout_conf().rate();
  CHECK_GE(dropout_rate, 0);
  CHECK_LT(dropout_rate, 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    EnrollOutputBn("random_mask");
  }
}

const PbMessage& DropoutOp::GetCustomizedConf() const { return op_conf().dropout_conf(); }

void DropoutOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  // CHECK_EQ(op_conf().dropout_conf().noise_shape().dim_size(),
  //          GetBlobDesc4BnInOp("in")->shape().NumAxes());
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    *GetBlobDesc4BnInOp("random_mask") = *GetBlobDesc4BnInOp("in");
    GetBlobDesc4BnInOp("random_mask")->set_data_type(DataType::kFloat);
  }
}

void DropoutOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  DropoutKernelConf* mut_dropout_conf = kernel_conf->mutable_dropout_conf();
  GetBlobDesc4BnInOp("in")->shape().ToProto(mut_dropout_conf->mutable_in());
  GetBlobDesc4BnInOp("in")->shape().ToProto(mut_dropout_conf->mutable_random_mask());
  GetBlobDesc4BnInOp("out")->shape().ToProto(mut_dropout_conf->mutable_out());
}

void DropoutOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  for (const auto& obn : output_bns()) {
    *HasBatchDim4BnInOp(obn) = *HasBatchDim4BnInOp(SoleIbn());
  }
}

void DropoutOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn("in").shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kDropoutConf, DropoutOp);

}  // namespace oneflow
