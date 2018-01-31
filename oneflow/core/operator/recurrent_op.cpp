#include "oneflow/core/operator/recurrent_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

void InferBasicRnnCellBlobDesc(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    int32_t hidden_size) {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t embedding_size = in_blob_desc->shape().Count(1);
  int64_t data_num = in_blob_desc->shape().At(0);
  *GetBlobDesc4BnInOp("plus_op_out") =
      BlobDesc(Shape({embedding_size, hidden_size}),
               JobDesc::Singleton()->DefaultDataType(), false, true,
               in_blob_desc->max_col_num());
  *GetBlobDesc4BnInOp("i2h_weight") =
      BlobDesc(Shape({hidden_size, embedding_size}));
  *GetBlobDesc4BnInOp("h2h_weight") =
      BlobDesc(Shape({hidden_size, hidden_size}));
  *GetBlobDesc4BnInOp("bias") = BlobDesc(Shape({1, hidden_size}));
  *GetBlobDesc4BnInOp("bias_multiplier") = BlobDesc(Shape({data_num, 1}));
}

}  //  namespace

void RecurrentOp::InitFromOpConf() {
  CHECK(op_conf().has_recurrent_conf());
  const RecurrentOpConf& conf = op_conf().recurrent_conf();
  EnrollInputBn("in");
  EnrollInputBn("rec_in");
  if (!conf.init_hidden().empty()) {
    CHECK(!conf.has_init_hidden_initializer());
    EnrollInputBn("h0");
  } else {
    EnrollModelBn("h0");
  }
  EnrollOutputBn("out");
  EnrollOutputBn("rec_out");

  if (conf.rnn_type_case() == RecurrentOpConf::kBasicRnnCell) {
    EnrollDataTmpBn("plus_op_out");
    EnrollModelBn("i2h_weight");
    EnrollModelBn("h2h_weight");
    EnrollModelBn("bias");
    EnrollModelTmpBn("bias_multiplier");
  } else if (conf.rnn_type_case() == RecurrentOpConf::kBasicLstmCell) {
    TODO();
  } else {
    UNEXPECTED_RUN();
  }
}

const PbMessage& RecurrentOp::GetSpecialConf() const {
  return op_conf().recurrent_conf();
}

void RecurrentOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const RecurrentOpConf& conf = op_conf().recurrent_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  DataType data_type = JobDesc::Singleton()->DefaultDataType();
  CHECK_EQ(in_blob_desc->data_type(), data_type);
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(in_blob_desc->has_col_num_field(), true);
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t hidden_size = conf.hidden_size();
  Shape h0_shape = Shape({data_num, hidden_size});
  if (!conf.init_hidden().empty()) {
    const BlobDesc* h0_blob_desc = GetBlobDesc4BnInOp("h0");
    CHECK_EQ(h0_blob_desc->data_type(), data_type);
    CHECK_EQ(h0_blob_desc->shape(), h0_shape);
    CHECK_EQ(h0_blob_desc->has_data_id_field(),
             in_blob_desc->has_data_id_field());
    CHECK_EQ(h0_blob_desc->max_col_num(), 1);
  } else {
    *GetBlobDesc4BnInOp("h0") = BlobDesc(h0_shape);
  }
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(hidden_size, parallel_ctx->parallel_num());
    hidden_size = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // out
  BlobDesc out_blob_desc = *in_blob_desc;
  out_blob_desc.mut_shape() = Shape({data_num, hidden_size});
  *GetBlobDesc4BnInOp("out") = out_blob_desc;
  // recurrent_out
  *GetBlobDesc4BnInOp("rec_out") = out_blob_desc;

  if (op_conf().recurrent_conf().rnn_type_case()
      == RecurrentOpConf::kBasicRnnCell) {
    InferBasicRnnCellBlobDesc(GetBlobDesc4BnInOp, hidden_size);
  } else if (op_conf().recurrent_conf().rnn_type_case()
             == RecurrentOpConf::kBasicLstmCell) {
    TODO();
  } else {
    UNEXPECTED_RUN();
  }
}

std::string RecurrentOp::ibn2lbn(const std::string& input_bn) const {
  if (input_bn == "rec_in") {
    return obn2lbn("rec_out");
  } else if (input_bn == "h0") {
    return op_conf().recurrent_conf().init_hidden();
  } else if (input_bn == "in") {
    return op_conf().recurrent_conf().in();
  } else {
    UNEXPECTED_RUN();
    return "";
  }
}

std::string RecurrentOp::obn2lbn(const std::string& output_bn) const {
  if (output_bn == "out") {
    return op_name() + "/" + op_conf().recurrent_conf().out();
  } else if (output_bn == "rec_out") {
    return op_name() + "/rec_" + op_conf().recurrent_conf().out();
  } else {
    UNEXPECTED_RUN();
    return "";
  }
}

REGISTER_OP(OperatorConf::kRecurrentConf, RecurrentOp);

}  // namespace oneflow
