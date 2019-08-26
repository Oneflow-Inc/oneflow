#include "oneflow/core/operator/extract_piece_slice_id_op.h"

namespace oneflow {

void ExtractPieceSliceIdOp::InitFromOpConf() {
  CHECK(op_conf().has_extract_piece_slice_id_conf());
  EnrollRepeatedInputBn("in", false);
  EnrollRepeatedOutputBn("out", false);
}

const PbMessage& ExtractPieceSliceIdOp::GetCustomizedConf() const {
  return this->op_conf().extract_piece_slice_id_conf();
}

void ExtractPieceSliceIdOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().extract_piece_slice_id_conf();
  const DataType data_type = GetBlobDesc4BnInOp(input_bns().Get(0))->data_type();
  FOR_RANGE(int32_t, i, 1, conf.in_size()) {
    CHECK_EQ(data_type, GetBlobDesc4BnInOp(input_bns().Get(i))->data_type());
  }
  CHECK_EQ(conf.in_size(), conf.out_size());
  FOR_RANGE(int32_t, i, 0, conf.in_size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(input_bns().Get(i));
    BlobDesc* out_i = GetBlobDesc4BnInOp(output_bns().Get(i));
    out_i->mut_shape() = Shape({in_i->shape().At(0)});
    out_i->set_data_type(DataType::kInt32);
    if (in_i->has_dim0_valid_num_field()) {
      out_i->set_has_dim0_valid_num_field(true);
      out_i->mut_dim0_inner_shape() = Shape({1, out_i->shape().At(0)});
    }
  }
}

void ExtractPieceSliceIdOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp(input_bns().Get(0))->data_type());
}

REGISTER_OP(OperatorConf::kExtractPieceSliceIdConf, ExtractPieceSliceIdOp);

}  // namespace oneflow
