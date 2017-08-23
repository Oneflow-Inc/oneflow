//#include "oneflow/core/operator/softmax_op.h"
//
// namespace oneflow {
//
// void SoftmaxOp::InitFromOpConf(const OperatorConf& op_conf) {
//  CHECK(op_conf.has_softmax_conf());
//  mut_op_conf() = op_conf;
//
//  EnrollInputBn("in");
//  EnrollOutputBn("out");
//  EnrollDataTmpBn("tmp");
//}
//
// const PbMessage& SoftmaxOp::GetSpecialConf() const {
//  return op_conf().softmax_conf();
//}
//
// void SoftmaxOp::InferBlobDesc4FwBlobs(
//    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
//    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
//  const std::vector<int64_t>& vec =
//      GetBlobDesc4BnInOp(SoleIbn())->shape().dim_vec();
//  CHECK_EQ(vec.size(), 2);
//  GetBlobDesc4BnInOp(SoleObn())->mut_shape() = Shape(vec);
//  GetBlobDesc4BnInOp(SoleDtbn())->mut_shape() = Shape({vec[0]});
//}
//
// REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);
//
//}  // namespace oneflow
