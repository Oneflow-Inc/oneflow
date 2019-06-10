#include "oneflow/core/operator/sink_tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SinkTickOp::InitFromOpConf() {
  CHECK(op_conf().has_sink_tick_conf());
  EnrollRepeatedInputBn("tick", false);
}

const PbMessage& SinkTickOp::GetCustomizedConf() const { return op_conf().sink_tick_conf(); }

void SinkTickOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Split(input_bns(), 0).Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_CPU_OP(OperatorConf::kSinkTickConf, SinkTickOp);

}  // namespace oneflow
