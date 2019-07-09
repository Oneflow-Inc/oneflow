#include "oneflow/core/operator/contiguous_instance_stack_op.h"

namespace oneflow {

void ContiguousInstanceStackOp::InitFromOpConf() {
  CHECK(op_conf().has_contiguous_instance_stack_conf());
  const int32_t in_size = op_conf().contiguous_instance_stack_conf().in_size();
  CHECK_GT(in_size, 0);
  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
}

void ContiguousInstanceStackOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0 = GetBlobDesc4BnInOp(output_bns().Get(0));
  CHECK(!in_0->has_dim1_valid_num_field());
  CHECK(!in_0->has_dim2_valid_num_field());
  const int32_t in_size = op_conf().contiguous_instance_stack_conf().in_size();
  FOR_RANGE(int32_t, i, 1, in_size) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(output_bns().Get(i));
    CHECK_EQ(in_0->shape(), in_i->shape());
    CHECK_EQ(in_0->data_type(), in_i->data_type());
    CHECK_EQ(in_0->has_dim0_valid_num_field(), in_i->has_dim0_valid_num_field());
    CHECK_EQ(in_0->has_instance_shape_field(), in_i->has_instance_shape_field());
    CHECK(!in_0->has_dim1_valid_num_field());
    CHECK(!in_0->has_dim2_valid_num_field());
  }
  BlobDesc* out = GetBlobDesc4BnInOp(SoleObn());
  std::vector<int64_t> out_dim_vec = in_0->shape().dim_vec();
  out_dim_vec.insert(out_dim_vec.begin(), in_size);
  out->mut_shape() = Shape(out_dim_vec);
  out->set_data_type(in_0->data_type());
  out->set_has_instance_shape_field(in_0->has_dim0_valid_num_field()
                                    || in_0->has_instance_shape_field());
}

REGISTER_OP(OperatorConf::kContiguousInstanceStackConf, ContiguousInstanceStackOp);

}  // namespace oneflow
