#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/input_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

void CheckOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.ctrl_in_op_name().empty());
  if (op_conf.input_conf().has_blob_conf()) {
    if (op_conf.input_conf().blob_conf().has_dim0_inner_shape()) { TODO(); }
    if (op_conf.input_conf().blob_conf().has_dim1_valid_num()) { TODO(); }
    if (op_conf.input_conf().blob_conf().has_dim2_valid_num()) { TODO(); }
  }
}

void CheckShape(const Shape& shape) {
  FOR_RANGE(int, i, 1, shape.NumAxes()) { CHECK_GT(shape.At(i), 0); }
}

bool GetSplitAxis(const InputBlobConf& input_blob_conf, size_t* split_axis) {
  if (input_blob_conf.has_split_axis()) {
    int64_t axis = input_blob_conf.split_axis();
    if (axis < 0) { axis += input_blob_conf.shape().dim_size(); }
    if (axis >= 0 && axis < input_blob_conf.shape().dim_size()) {
      *split_axis = axis;
      return true;
    }
  } else if (input_blob_conf.broadcast() == false && input_blob_conf.has_batch_dim()) {
    *split_axis = 0;
    return true;
  }
  return false;
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
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  CheckOpConf(op_conf());
  const auto& conf = op_conf().input_conf().blob_conf();
  out_blob_desc->mut_shape() = Shape(conf.shape());
  CheckShape(out_blob_desc->shape());
  if (out_blob_desc->mut_shape().At(0) == -1) {
    CHECK_EQ(record_piece_size % parallel_ctx->parallel_num(), 0);
    out_blob_desc->mut_shape().Set(0, record_piece_size);
  } else {
    CHECK_GT(out_blob_desc->mut_shape().At(0), 0);
  }
  if (conf.has_data_type()) {
    out_blob_desc->set_data_type(conf.data_type());
  } else {
    out_blob_desc->set_data_type(GlobalJobDesc().DefaultDataType());
  }
  out_blob_desc->set_has_dim0_valid_num_field(conf.has_dim0_valid_num());
  size_t split_axis = 0;
  if (GetSplitAxis(conf, &split_axis)) {
    BalancedSplitter bs(out_blob_desc->shape().At(split_axis), parallel_ctx->parallel_num());
    out_blob_desc->mut_shape().Set(split_axis, bs.At(parallel_ctx->parallel_id()).size());
  }
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
