#include <string>
#include <vector>
#include "operator/innerproduct_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"
#include "common/balanced_splitter.h"

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
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  Shape* in_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  uint32_t out_num = GetUInt32FromSpecialConf("out_num");
  if (policy == kModelParallel) {
    BalancedSplitter splitter(out_num, parallel_num);
    out_num = splitter.At(parallel_id).size();
  }
  int32_t axis = GetInt32FromSpecialConf("axis");

  // output bn
  Shape* out_shape_ptr = GetShapePtr4BnInOp(SoleObn());
  *out_shape_ptr = *in_shape_ptr;
  out_shape_ptr->Set(axis, out_num);
  if (axis < 0) {
    for (int32_t i = axis + 1; i < 0; ++i) {
      out_shape_ptr->Set(i, 1);
    }
  } else {
    for (int32_t i = axis + 1; i < out_shape_ptr->NumAxes(); ++i) {
      out_shape_ptr->Set(i, 1);
    }
  }

  // model bn
  CHECK_EQ(model_bns().size(), 2);
  Shape* weight_shape_ptr = GetShapePtr4BnInOp(model_bns().at(0));
  Shape* bias_shape_ptr = GetShapePtr4BnInOp(model_bns().at(1));
  *weight_shape_ptr = Shape(std::vector<int64_t>(in_shape_ptr->NumAxes(), 1));
  weight_shape_ptr->Set(0, out_num);
  weight_shape_ptr->Set(1, in_shape_ptr->Count(axis));

  *bias_shape_ptr = Shape(std::vector<int64_t>(in_shape_ptr->NumAxes(), 1));
  bias_shape_ptr->Set(1, out_num);

  // model tmp bn
  CHECK_EQ(model_tmp_bns().size(), 1);
  Shape* bias_multiplier_shape_ptr = GetShapePtr4BnInOp(model_tmp_bns().at(0));
  *bias_multiplier_shape_ptr = Shape(
      std::vector<int64_t>(bias_shape_ptr->NumAxes(), 1));
  bias_multiplier_shape_ptr->Set(0, in_shape_ptr->At(0));
}

REGISTER_OP(OperatorConf::kInnerproductConf, InnerProductOp);

}  // namespace oneflow
