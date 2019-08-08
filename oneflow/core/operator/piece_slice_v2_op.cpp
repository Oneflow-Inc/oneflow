#include "oneflow/core/operator/piece_slice_v2_op.h"

namespace oneflow {

void PieceSliceV2Op::InitFromOpConf() {
  CHECK(op_conf().has_piece_slice_v2_conf());
  EnrollInputBn("in");
  const int32_t out_size = op_conf().piece_slice_v2_conf().out_size();
  CHECK_GT(out_size, 0);
  EnrollRepeatedOutputBn("out", out_size);
}

void PieceSliceV2Op::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp(SoleIbn());
  CHECK(!in->has_dim0_valid_num_field());
  const int32_t out_size = op_conf().piece_slice_v2_conf().out_size();
  // out_size should be equal to in->shape().At(0), but record_piece_size is set to 1 in op_graph,
  // so we use global device piece size to do this check
  CHECK_EQ(out_size, Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(*parallel_ctx));

  const bool uncontiguous_varing_in =
      in->has_dim1_valid_num_field() || in->has_dim2_valid_num_field();
  const bool contiguous_varing_in = in->has_instance_shape_field();
  FOR_RANGE(int32_t, i, 0, out_size) {
    BlobDesc* out_i = GetBlobDesc4BnInOp(output_bns().Get(i));
    out_i->mut_shape() =
        Shape(std::vector<int64_t>(in->shape().dim_vec().begin() + 1, in->shape().dim_vec().end()));
    out_i->set_data_type(in->data_type());
    if (uncontiguous_varing_in) {
      CHECK(!contiguous_varing_in);
      out_i->set_has_dim0_valid_num_field(true);
      out_i->mut_dim0_inner_shape() = Shape({1, out_i->shape().At(0)});
      out_i->set_has_dim1_valid_num_field(in->has_dim2_valid_num_field());
      out_i->set_has_dim2_valid_num_field(false);
    } else if (contiguous_varing_in) {
      CHECK(!uncontiguous_varing_in);
      out_i->set_has_dim0_valid_num_field(true);
      out_i->mut_dim0_inner_shape() = Shape({1, out_i->shape().At(0)});
      out_i->set_has_instance_shape_field(true);
    }
  }
}

REGISTER_OP(OperatorConf::kPieceSliceV2Conf, PieceSliceV2Op);

}  // namespace oneflow
