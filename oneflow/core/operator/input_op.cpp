#include "oneflow/core/operator/input_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

void CheckOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.ctrl_in_op_name().empty());
  if (op_conf.input_conf().blob_conf().has_dim0_inner_shape()) { TODO(); }
  if (op_conf.input_conf().blob_conf().has_dim1_valid_num()) { TODO(); }
  if (op_conf.input_conf().blob_conf().has_dim2_valid_num()) { TODO(); }
}

void CheckShape(const Shape& shape) {
  FOR_RANGE(int, i, 1, shape.NumAxes()) { CHECK_GT(shape.At(i), 0); }
}

}  // namespace

void InputOp::InitFromOpConf() {
  CHECK(op_conf().has_input_conf());
  if (op_conf().input_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

const PbMessage& InputOp::GetCustomizedConf() const { return op_conf().input_conf(); }

void InputOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, int64_t record_piece_size) const {
  CheckOpConf(op_conf());
  const auto& conf = op_conf().input_conf().blob_conf();
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(conf.shape());
  CheckShape(out_blob_desc->shape());
  if (out_blob_desc->mut_shape().At(0) == -1) {
    CHECK_EQ(record_piece_size % parallel_ctx->parallel_num(), 0);
    out_blob_desc->mut_shape().Set(0, record_piece_size / parallel_ctx->parallel_num());
  } else {
    CHECK_GT(out_blob_desc->mut_shape().At(0), 0);
  }
  if (conf.has_data_type()) {
    out_blob_desc->set_data_type(conf.data_type());
  } else {
    out_blob_desc->set_data_type(GlobalJobDesc().DefaultDataType());
  }
  out_blob_desc->set_has_dim1_valid_num_field(conf.dim0_valid_num());
}

void InputOp::InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = op_conf().input_conf().blob_conf().has_batch_dim();
}

void InputOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  int64_t num_axes = op_conf().input_conf().blob_conf().shape().dim_size();
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kInputConf, InputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kInputConf, 1);

}  // namespace oneflow
