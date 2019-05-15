#include "oneflow/core/operator/parallel_cast_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ParallelCastOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  const ParallelCastOpConf& conf = op_conf().parallel_cast_conf();
  if (conf.has_split_parallel()) {
    SbpSignatureBuilder()
        .Split(input_bns(), conf.split_parallel().axis())
        .Split(output_bns(), conf.split_parallel().axis())
        .Build(sbp_signature);
  } else if (conf.has_broadcast_parallel()) {
    SbpSignatureBuilder().Broadcast(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
  } else {
    UNIMPLEMENTED();
  }
}

REGISTER_OP(OperatorConf::kParallelCastConf, ParallelCastOp);

}  // namespace oneflow
