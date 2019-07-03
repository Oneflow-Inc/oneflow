#include "oneflow/core/operator/piece_slice_op.h"

namespace oneflow {

void PieceSliceOp::InitFromOpConf() {
  CHECK(op_conf().has_piece_slice_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void PieceSliceOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp(SoleIbn());
  BlobDesc* out = GetBlobDesc4BnInOp(SoleObn());
  // FIX ME: slice_num should be inferred instead of setting by hand
  // we set it by hand in order to infer time_shape in op_graph
  CHECK_EQ(GetSliceNum(), in->shape().At(0));
  CHECK(!in->has_dim0_valid_num_field());

  out->mut_shape() =
      Shape(std::vector<int64_t>(in->shape().dim_vec().begin() + 1, in->shape().dim_vec().end()));
  out->set_data_type(in->data_type());
  const bool uncontiguous_varing_instance =
      in->has_dim1_valid_num_field() || in->has_dim2_valid_num_field();
  const bool contiguous_varing_instance = in->has_instance_shape_field();
  if (uncontiguous_varing_instance) {
    CHECK(!contiguous_varing_instance);
    out->set_has_dim0_valid_num_field(true);
    out->mut_dim0_inner_shape() = Shape({1, out->shape().At(0)});
    out->set_has_dim1_valid_num_field(in->has_dim2_valid_num_field());
    out->set_has_dim2_valid_num_field(false);
  } else if (contiguous_varing_instance) {
    CHECK(!uncontiguous_varing_instance);
    out->set_has_dim0_valid_num_field(true);
    out->mut_dim0_inner_shape() = Shape({1, out->shape().At(0)});
    out->set_has_instance_shape_field(in->shape().NumAxes() > 2);
  }
}

void PieceSliceOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  std::vector<int64_t> dim_vec(GetTimeShape4BnInOp("in")->dim_vec());
  dim_vec.push_back(GetSliceNum());
  *time_shape = Shape(dim_vec);
}

const int32_t PieceSliceOp::GetSliceNum() const {
  CHECK(op_conf().has_piece_slice_conf());
  return op_conf().piece_slice_conf().slice_num();
}

REGISTER_OP(OperatorConf::kPieceSliceConf, PieceSliceOp);

}  // namespace oneflow
