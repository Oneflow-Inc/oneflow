#include "oneflow/core/operator/concat_op.h"

namespace oneflow {

void ConcatOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_concat_conf());
  mut_op_conf() = op_conf;

  for (int i = 0; i < op_conf.concat_conf().in_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    CHECK(ibn2lbn_.emplace(ibn, op_conf.concat_conf().in(i)).second);
    EnrollInputBn(ibn);
  }
  EnrollOutputBn("out");
}

const PbMessage& ConcatOp::GetSpecialConf() const {
  return op_conf().concat_conf();
}

void ConcatOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  std::vector<int64_t> vec = GetShapePtr4BnInOp(input_bns().at(0))->dim_vec();
  for (size_t ibn_idx = 1; ibn_idx < input_bns().size(); ++ibn_idx) {
    Shape* ib_shape = GetShapePtr4BnInOp(input_bns().at(ibn_idx));
    int32_t concat_axis = op_conf().concat_conf().axis();
    for (int64_t j = 0; j < ib_shape->NumAxes(); ++j) {
      if (j == concat_axis || j == concat_axis + ib_shape->NumAxes()) {
        vec[j] += ib_shape->At(j);
      } else {
        CHECK_EQ(vec[j], ib_shape->At(j));
      }
    }
  }
  CHECK_EQ(vec.size(), GetShapePtr4BnInOp(input_bns().at(0))->NumAxes());
  *GetShapePtr4BnInOp(SoleObn()) = Shape(vec);
}

REGISTER_OP(OperatorConf::kConcatConf, ConcatOp);

} // namespace oneflow
