#include "oneflow/core/operator/boxing_concat_op.h"

namespace oneflow {

void BoxingConcatOp::InitFromOpConf() {
  const BoxingConcatOpConf& conf = op_conf().boxing_concat_conf();
  EnrollRepeatedInputBn("in", conf.in_num(), false);
  EnrollOutputBn("out", false);
}

const PbMessage& BoxingConcatOp::GetCustomizedConf() const {
  return op_conf().boxing_concat_conf();
}

void BoxingConcatOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const BoxingConcatOpConf& conf = op_conf().boxing_concat_conf();
  const BlobDesc* in_0 = GetBlobDesc4BnInOp(GenRepeatedBn("in", 0));
  bool has_dim0_valid_num_field = in_0->has_dim0_valid_num_field();
  int64_t concat_dim_size = in_0->shape().At(conf.axis());
  FOR_RANGE(int64_t, i, 1, conf.in_num()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    if (in_i->has_dim0_valid_num_field()) { has_dim0_valid_num_field = true; }
    CHECK_EQ(in_i->shape().NumAxes(), in_0->shape().NumAxes());
    FOR_RANGE(int64_t, axis, 0, in_0->shape().NumAxes()) {
      if (axis == conf.axis()) {
        concat_dim_size += in_i->shape().At(axis);
      } else {
        CHECK_EQ(in_i->shape().At(axis), in_0->shape().At(axis));
      }
    }
  }
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_has_dim0_valid_num_field(has_dim0_valid_num_field);
  std::vector<int64_t> out_shape_dim_vec = in_0->shape().dim_vec();
  out_shape_dim_vec[conf.axis()] = concat_dim_size;
  out->mut_shape() = Shape(out_shape_dim_vec);
  out->mut_dim0_inner_shape() = Shape({1, out_shape_dim_vec.front()});
}

LogicalBlobId BoxingConcatOp::ibn2lbi(const std::string& input_bn) const {
  return op_conf().boxing_concat_conf().lbi();
}

LogicalBlobId BoxingConcatOp::obn2lbi(const std::string& output_bn) const {
  return op_conf().boxing_concat_conf().lbi();
}
REGISTER_OP(OperatorConf::kBoxingConcatConf, BoxingConcatOp);

}  // namespace oneflow
