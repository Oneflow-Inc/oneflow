//#include "oneflow/core/operator/innerproduct_op.h"
//#include "oneflow/core/common/balanced_splitter.h"
//
// namespace oneflow {
//
// void InnerProductOp::InitFromOpConf(const OperatorConf& op_conf) {
//  CHECK(op_conf.has_innerproduct_conf());
//  mut_op_conf() = op_conf;
//
//  EnrollInputBn("in");
//  EnrollOutputBn("out");
//  EnrollModelBn("weight");
//
//  if (GetBoolFromSpecialConf("has_bias_term")) {
//    EnrollModelBn("bias");
//    EnrollModelTmpBn("bias_multiplier");
//  }
//}
//
// const PbMessage& InnerProductOp::GetSpecialConf() const {
//  return op_conf().innerproduct_conf();
//}
//
// void InnerProductOp::InferBlobDesc4FwBlobs(
//    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
//    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
//  const Shape& in_shape = GetBlobDesc4BnInOp(SoleIbn())->shape();
//  int32_t out_num = GetInt32FromSpecialConf("out_num");
//  if (policy == kModelParallel) {
//    BalancedSplitter splitter(out_num, parallel_num);
//    out_num = splitter.At(parallel_id).size();
//  }
//
//  // output bn
//  GetBlobDesc4BnInOp(SoleObn())->mut_shape() = Shape({in_shape.At(0),
//  out_num});
//
//  // model bn
//  GetBlobDesc4BnInOp("weight")->mut_shape() =
//      Shape({out_num, in_shape.Count(1)});
//
//  if (GetBoolFromSpecialConf("has_bias_term")) {
//    // model bn
//    GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({1, out_num});
//
//    // model tmp bn
//    CHECK_EQ(model_tmp_bns().size(), 1);
//    GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() =
//        Shape({in_shape.At(0), 1});
//  }
//}
//
// REGISTER_OP(OperatorConf::kInnerproductConf, InnerProductOp);
//
//}  // namespace oneflow
