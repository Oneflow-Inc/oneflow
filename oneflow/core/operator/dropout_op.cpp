#include "oneflow/core/operator/dropout_op.h"

namespace oneflow {

void DropoutOp::InitFromOpConf() {
  if (op_conf().dropout_conf().has_noise_shape()) { TODO(); }
  double dropout_rate = op_conf().dropout_conf().rate();
  CHECK_GE(dropout_rate, 0);
  CHECK_LT(dropout_rate, 1);
  EnrollInputBn("x");
  EnrollOutputBn("y");
  if (Global<JobDesc>::Get()->IsTrain()) {
    EnrollDataTmpBn("random_mask");
  } else if (Global<JobDesc>::Get()->IsPredict()
             && Global<JobDesc>::Get()
                    ->other_conf()
                    .predict_conf()
                    .has_tmp_split_fw_bw_train_conf()) {
    EnrollOutputBn("random_mask");
  }
}

const PbMessage& DropoutOp::GetCustomizedConf() const { return op_conf().dropout_conf(); }

void DropoutOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  // CHECK_EQ(op_conf().dropout_conf().noise_shape().dim_size(),
  //          GetBlobDesc4BnInOp("x")->shape().NumAxes());
  *GetBlobDesc4BnInOp("y") = *GetBlobDesc4BnInOp("x");
  if (Global<JobDesc>::Get()->IsTrain()
      || (Global<JobDesc>::Get()->IsPredict()
          && Global<JobDesc>::Get()
                 ->other_conf()
                 .predict_conf()
                 .has_tmp_split_fw_bw_train_conf())) {
    *GetBlobDesc4BnInOp("random_mask") = *GetBlobDesc4BnInOp("x");
    GetBlobDesc4BnInOp("random_mask")->set_data_type(DataType::kFloat);
  }
}

void DropoutOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  DropoutKernelConf* mut_dropout_conf = kernel_conf->mutable_dropout_conf();
  GetBlobDesc4BnInOp("x")->shape().ToProto(mut_dropout_conf->mutable_in());
  GetBlobDesc4BnInOp("x")->shape().ToProto(mut_dropout_conf->mutable_random_mask());
  GetBlobDesc4BnInOp("y")->shape().ToProto(mut_dropout_conf->mutable_out());
}

REGISTER_OP(OperatorConf::kDropoutConf, DropoutOp);

}  // namespace oneflow
