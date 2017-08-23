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

void ConcatOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  std::vector<int64_t> vec =
      GetBlobDesc4BnInOp(input_bns().at(0))->shape().dim_vec();
  for (size_t ibn_idx = 1; ibn_idx < input_bns().size(); ++ibn_idx) {
    const Shape& ib_shape =
        GetBlobDesc4BnInOp(input_bns().at(ibn_idx))->shape();
    int32_t concat_axis = op_conf().concat_conf().axis();
    for (int64_t j = 0; j < ib_shape.NumAxes(); ++j) {
      if (j == concat_axis || j == concat_axis + ib_shape.NumAxes()) {
        vec[j] += ib_shape.At(j);
      } else {
        CHECK_EQ(vec[j], ib_shape.At(j));
      }
    }
  }
  CHECK_EQ(vec.size(),
           GetBlobDesc4BnInOp(input_bns().at(0))->shape().NumAxes());
  GetBlobDesc4BnInOp(SoleObn())->mut_shape() = Shape(vec);
}

REGISTER_OP(OperatorConf::kConcatConf, ConcatOp);

}  // namespace oneflow
