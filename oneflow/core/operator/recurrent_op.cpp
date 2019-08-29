#include "oneflow/core/operator/recurrent_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void RecurrentOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollInputBn("rec_in");
  if (!GetValFromCustomizedConf<std::string>("init_hidden").empty()) {
    CHECK(!GetValFromCustomizedConf<bool>("has_init_hidden_initializer"));
    EnrollInputBn("h0");
  } else if (GetValFromCustomizedConf<bool>("is_init_hidden_trainable")) {
    EnrollTmpBn("h0");
  } else {
    EnrollConstBufBn("h0");
  }
  EnrollOutputBn("out");
  EnrollOutputBn("rec_out");
  VirtualInitFromOpConf();
}

void RecurrentOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  DataType data_type = GlobalJobDesc().DefaultDataType();
  CHECK_EQ(in_blob_desc->data_type(), data_type);
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(in_blob_desc->has_col_num_field(), true);
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t hidden_size = GetValFromCustomizedConf<int32_t>("hidden_size");
  Shape h0_shape = Shape({data_num, hidden_size});
  if (!GetValFromCustomizedConf<std::string>("init_hidden").empty()) {
    const BlobDesc* h0_blob_desc = GetBlobDesc4BnInOp("h0");
    CHECK_EQ(h0_blob_desc->data_type(), data_type);
    CHECK_EQ(h0_blob_desc->shape(), h0_shape);
    CHECK_EQ(h0_blob_desc->has_data_id_field(), in_blob_desc->has_data_id_field());
    CHECK_EQ(h0_blob_desc->max_col_num(), 1);
  } else {
    // *GetBlobDesc4BnInOp("h0") = BlobDesc(h0_shape);
    TODO();
  }
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(hidden_size, parallel_ctx->parallel_num());
    hidden_size = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape({data_num, hidden_size});
  // recurrent_out
  BlobDesc* rec_out_blob_desc = GetBlobDesc4BnInOp("rec_out");
  *rec_out_blob_desc = *out_blob_desc;
  if (parallel_ctx->policy() == kDataParallel) { rec_out_blob_desc->set_max_col_num(1); }

  VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

LogicalBlobId RecurrentOp::ibn2lbi(const std::string& input_bn) const {
  if (input_bn == "rec_in") {
    return obn2lbi("rec_out");
  } else if (input_bn == "h0") {
    return GenLogicalBlobId(GetValFromCustomizedConf<std::string>("init_hidden"));
  } else if (input_bn == "in") {
    return GenLogicalBlobId(GetValFromCustomizedConf<std::string>("in"));
  } else {
    UNIMPLEMENTED();
  }
}

LogicalBlobId RecurrentOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  if (output_bn == "out") {
    ret.set_blob_name(GetValFromCustomizedConf<std::string>("out"));
  } else if (output_bn == "rec_out") {
    ret.set_blob_name("/rec_" + GetValFromCustomizedConf<std::string>("out"));
  } else {
    UNIMPLEMENTED();
  }
  return ret;
}

}  // namespace oneflow
