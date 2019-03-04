#include "oneflow/core/operator/reshape_instance_shape_to_dim0_op.h"

namespace oneflow {

void ReshapeInstanceShapeToDim0Op::InitFromOpConf() {
  CHECK(op_conf().has_reshape_instance_shape_to_dim0_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ReshapeInstanceShapeToDim0Op::GetCustomizedConf() const {
  return op_conf().reshape_instance_shape_to_dim0_conf();
}

void ReshapeInstanceShapeToDim0Op::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ReshapeInstanceShapeToDim0OpConf& conf = op_conf().reshape_instance_shape_to_dim0_conf();
  for (const int64_t dim : conf.shape().dim()) { CHECK_GE(dim, 1); }
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  CHECK(in->has_instance_shape_field());
  const Shape out_instance_shape(conf.shape());
  if (out_instance_shape.elem_cnt() > 0) {
    CHECK_EQ(in->shape().elem_cnt() % out_instance_shape.elem_cnt(), 0);
    const int64_t out_dim0 = in->shape().elem_cnt() / out_instance_shape.elem_cnt();
    std::vector<int64_t> out_shape_dim_vec;
    out_shape_dim_vec.push_back(out_dim0);
    out_shape_dim_vec.insert(out_shape_dim_vec.end(), out_instance_shape.dim_vec().cbegin(),
                             out_instance_shape.dim_vec().cend());
    out->mut_shape() = Shape(out_shape_dim_vec);
    out->set_has_dim0_valid_num_field(true);
    out->mut_dim0_inner_shape() = Shape({1, out_dim0});
  } else {
    const int64_t out_dim0 = in->shape().elem_cnt();
    std::vector<int64_t> out_shape_dim_vec;
    out_shape_dim_vec.push_back(out_dim0);
    out->mut_shape() = Shape(out_shape_dim_vec);
    out->set_has_dim0_valid_num_field(true);
    out->mut_dim0_inner_shape() = Shape({1, out_dim0});
  }
  out->set_data_type(in->data_type());
}

REGISTER_OP(OperatorConf::kReshapeInstanceShapeToDim0Conf, ReshapeInstanceShapeToDim0Op);

}  // namespace oneflow
