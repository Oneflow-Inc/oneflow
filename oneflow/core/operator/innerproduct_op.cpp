#include "oneflow/core/operator/innerproduct_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void InnerProductOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_innerproduct_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");

  EnrollModelBn("weight");
  EnrollModelBn("bias");
  EnrollModelTmpBn("bias_multiplier");
}

const PbMessage& InnerProductOp::GetSpecialConf() const {
  return op_conf().innerproduct_conf();
}

void InnerProductOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    int64_t parallel_id,
    int64_t parallel_num) const {
  Shape* in_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  int32_t out_num = GetInt32FromSpecialConf("out_num");
  if (policy == kModelParallel) {
    BalancedSplitter splitter(out_num, parallel_num);
    out_num = splitter.At(parallel_id).size();
  }

  // output bn
  Shape* out_shape_ptr = GetShapePtr4BnInOp(SoleObn());
  *out_shape_ptr = Shape({in_shape_ptr->At(0), out_num});

  // model bn
  CHECK_EQ(model_bns().size(), 2);
  Shape* weight_shape_ptr = GetShapePtr4BnInOp("weight");
  Shape* bias_shape_ptr = GetShapePtr4BnInOp("bias");
  *weight_shape_ptr = Shape({out_num, in_shape_ptr->Count(1)});
  *bias_shape_ptr = Shape({out_num});

  // model tmp bn
  CHECK_EQ(model_tmp_bns().size(), 1);
  Shape* bias_multiplier_shape_ptr = GetShapePtr4BnInOp("bias_multiplier");
  *bias_multiplier_shape_ptr = Shape({in_shape_ptr->At(0), 1});
}

REGISTER_OP(OperatorConf::kInnerproductConf, InnerProductOp);

}  // namespace oneflow
