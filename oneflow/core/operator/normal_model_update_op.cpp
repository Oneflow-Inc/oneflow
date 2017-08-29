//#include "oneflow/core/operator/normal_model_update_op.h"
//
// namespace oneflow {
//
// void NormalModelUpdateOp::InitFromOpConf(const OperatorConf& op_conf) {
//  CHECK(op_conf.has_normal_mdupdt_conf());
//  mut_op_conf() = op_conf;
//
//  EnrollInputBn("model_diffs", false);
//  EnrollOutputBn("model", false);
//}
//
// const PbMessage& NormalModelUpdateOp::GetSpecialConf() const {
//  return op_conf().normal_mdupdt_conf();
//}
//
// REGISTER_OP(OperatorConf::kNormalMdupdtConf, NormalModelUpdateOp);
//
//}  // namespace oneflow
