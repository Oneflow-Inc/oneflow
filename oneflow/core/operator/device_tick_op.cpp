#include "oneflow/core/operator/device_tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void DeviceTickOp::InitFromOpConf() {
  EnrollRepeatedInputBn("tick", false);
  EnrollOutputBn("out", false);
}

Maybe<void> DeviceTickOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

Maybe<void> DeviceTickOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> DeviceTickOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kDeviceTickConf, 2);
REGISTER_OP(OperatorConf::kDeviceTickConf, DeviceTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kDeviceTickConf);

}  // namespace oneflow
