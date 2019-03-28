#include "oneflow/core/operator/normalization_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void NormalizationOp::InitFromOpConf() {
  const NormalizationOpConf& conf = op_conf().normalization_conf();
#ifdef WITH_CUDA
  if (DevIsGpuAndEnableCudnn()) { CHECK_GE(conf.epsilon(), CUDNN_BN_MIN_EPSILON); }
#endif
  CHECK_GE(conf.momentum(), 0);
  CHECK_LE(conf.momentum(), 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollForwardModelBn("moving_mean");
  EnrollForwardModelBn("moving_variance");
  if (conf.center()) {
    EnrollModelBn("beta");
  } else {
    EnrollDataTmpBn("beta_diff");
  }
  if (conf.scale()) {
    EnrollModelBn("gamma");
  } else {
    EnrollDataTmpBn("gamma_diff");
  }
  EnrollDataTmpBn("mean");
  EnrollDataTmpBn("inv_variance");
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NormalizationOpConf& conf = op_conf().normalization_conf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const DataType data_type = in->data_type();
  *GetBlobDesc4BnInOp("out") = *in;


}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
