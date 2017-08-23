//#include "oneflow/core/operator/relu_op.h"
//
// namespace oneflow {
//
// void ReluOp::InitFromOpConf(const OperatorConf& op_conf) {
//  CHECK(op_conf.has_relu_conf());
//  mut_op_conf() = op_conf;
//
//  EnrollInputBn("in");
//  EnrollOutputBn("out");
//}
//
// const PbMessage& ReluOp::GetSpecialConf() const {
//  return op_conf().relu_conf();
//}
//
// void ReluOp::InferBlobDesc4FwBlobs(
//    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
//    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
//  *GetBlobDesc4BnInOp(SoleObn()) = *GetBlobDesc4BnInOp(SoleIbn());
//}
//
// REGISTER_OP(OperatorConf::kReluConf, ReluOp);
//
//}  // namespace oneflow
