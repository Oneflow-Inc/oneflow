//#include "oneflow/core/operator/loss_record_op.h"
//
// namespace oneflow {
//
// void LossRecordOp::InitFromOpConf(const OperatorConf& op_conf) {
//  CHECK(op_conf.has_loss_record_conf());
//  mut_op_conf() = op_conf;
//  EnrollInputBn("loss_acc");
//}
//
// const PbMessage& LossRecordOp::GetSpecialConf() const {
//  return op_conf().loss_record_conf();
//}
//
// REGISTER_OP(OperatorConf::kLossRecordConf, LossRecordOp);
//
//}  // namespace oneflow
