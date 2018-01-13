#include "oneflow/core/operator/recurrent_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

void InferBasicRnnCellBlobDesc(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    int32_t hidden_size, bool has_bias_term) {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t embedding_size = in_blob_desc->shape().At(1);
  int64_t piece_size = in_blob_desc->shape().At(0);
  BlobDesc data_tmp_blob_desc =
      BlobDesc(Shape({embedding_size, hidden_size}),
               JobDesc::Singleton()->DefaultDataType(), false, false,
               in_blob_desc->max_col_num());
  *GetBlobDesc4BnInOp("in_ip_op_out") = data_tmp_blob_desc;
  *GetBlobDesc4BnInOp("hidden_ip_op_out") = data_tmp_blob_desc;
  *GetBlobDesc4BnInOp("plus_op_out") = data_tmp_blob_desc;
  *GetBlobDesc4BnInOp("f_op_out") = data_tmp_blob_desc;

  *GetBlobDesc4BnInOp("in_ip_op_weight") =
      BlobDesc(Shape({hidden_size, embedding_size}));
  *GetBlobDesc4BnInOp("hidden_ip_op_weight") =
      BlobDesc(Shape({hidden_size, hidden_size}));
  if (has_bias_term) {
    *GetBlobDesc4BnInOp("bias") = BlobDesc(Shape({1, hidden_size}));
    *GetBlobDesc4BnInOp("bias_multiplier") = BlobDesc(Shape({piece_size, 1}));
  }
}

}  //  namespace

void RecurrentOp::InitFromOpConf() {
  CHECK(op_conf().has_recurrent_conf());
  const RecurrentOpConf& conf = op_conf().recurrent_conf();
  EnrollInputBn("in");
  EnrollInputBn("ht_1");
  if (!conf.init_hidden().empty()) {
    CHECK(!conf.has_init_hidden_initializer());
    EnrollInputBn("h0");
  } else {
    EnrollModelBn("h0");
  }
  EnrollOutputBn("ht");
  EnrollOutputBn("out");

  if (conf.rnn_type_case() == RecurrentOpConf::kBasicRnnCell) {
    EnrollDataTmpBn("in_ip_op_out");
    EnrollDataTmpBn("hidden_ip_op_out");
    EnrollDataTmpBn("plus_op_out");
    EnrollDataTmpBn("f_op_out");
    EnrollModelBn("in_ip_op_weight");
    EnrollModelBn("hidden_ip_op_weight");
    if (conf.has_bias_term()) {
      EnrollModelBn("bias");
      EnrollModelTmpBn("bias_multiplier");
    }
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
  int64_t piece_size = in_blob_desc->shape().At(0);
  int32_t hidden_size = conf.hidden_size();
  Shape h0_shape = Shape({piece_size, hidden_size});
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
  // ht -- for recurrent edge
  BlobDesc* ht_blob_desc = GetBlobDesc4BnInOp("ht");
  ht_blob_desc->mut_shape() = Shape({piece_size, hidden_size});
  ht_blob_desc->set_data_type(data_type);
  ht_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());
  ht_blob_desc->set_max_col_num(in_blob_desc->max_col_num());
  // out
  *GetBlobDesc4BnInOp("out") = *ht_blob_desc;

  if (op_conf().recurrent_conf().rnn_type_case()
      == RecurrentOpConf::kBasicRnnCell) {
    InferBasicRnnCellBlobDesc(GetBlobDesc4BnInOp, hidden_size,
                              conf.has_bias_term());
  } else if (op_conf().recurrent_conf().rnn_type_case()
             == RecurrentOpConf::kBasicLstmCell) {
    TODO();
  } else {
    UNEXPECTED_RUN();
  }
}

std::string RecurrentOp::ibn2lbn(const std::string& input_bn) const {
  if (input_bn == "ht_1") {
    return obn2lbn("ht");
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
  return op_name() + "/" + op_conf().recurrent_conf().out();
}

REGISTER_OP(OperatorConf::kRecurrentConf, RecurrentOp);

}  // namespace oneflow
