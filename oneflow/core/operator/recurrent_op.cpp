#include "oneflow/core/operator/recurrent_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void RecurrentOp::InitFromOpConf() {
  CHECK(op_conf().has_recurrent_conf());
  const RecurrentOpConf& conf = op_conf().recurrent_conf();
  EnrollInputBn("in");
  EnrollInputBn("ht_1");
  if (!conf.init_hidden().empty()) { EnrollInputBn("h0"); }
  EnrollOutputBn("ht");

  if (conf.rnn_type_case() == RecurrentOpConf::kBasicRnnCell) {
    EnrollDataTmpBn("in_ip_op_out");
    EnrollDataTmpBn("hidden_ip_op_out");
    EnrollDataTmpBn("plus_op_out");
    EnrollDataTmpBn("f_op_out");
    EnrollModelBn("in_ip_op_weight");
    EnrollModelBn("hidden_ip_op_weight");
    if (GetBoolFromSpecialConf("has_bias_term")) {
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
  auto data_type = JobDesc::Singleton()->DefaultDataType();
  CHECK_EQ(in_blob_desc->data_type(), data_type);
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 2);
  if (!conf.init_hidden().empty()) {
    const BlobDesc* h0_blob_desc = GetBlobDesc4BnInOp("h0");
    CHECK_EQ(h0_blob_desc->data_type(), data_type);
    CHECK_EQ(in_blob_desc->shape(), h0_blob_desc->shape());
  }
  int32_t embedding_size = in_blob_desc->shape().At(1);
  int32_t piece_size = in_blob_desc->shape().At(0);
  int32_t hidden_size = conf.hidden_size();
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(hidden_size, parallel_ctx->parallel_num());
    hidden_size = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // ht
  BlobDesc* ht_blob_desc = GetBlobDesc4BnInOp("ht");
  ht_blob_desc->mut_shape() = Shape({piece_size, hidden_size});
  ht_blob_desc->set_data_type(data_type);
  ht_blob_desc->set_has_data_id(in_blob_desc->has_data_id());

  if (op_conf().recurrent_conf().rnn_type_case()
      == RecurrentOpConf::kBasicRnnCell) {
    *GetBlobDesc4BnInOp("in_ip_op_out") = *ht_blob_desc;
    *GetBlobDesc4BnInOp("hidden_ip_op_out") = *ht_blob_desc;
    *GetBlobDesc4BnInOp("plus_op_out") = *ht_blob_desc;
    *GetBlobDesc4BnInOp("f_op_out") = *ht_blob_desc;

    BlobDesc weight_blob_desc =
        BlobDesc(Shape({hidden_size, embedding_size}), data_type, false);
    *GetBlobDesc4BnInOp("in_ip_op_weight") = weight_blob_desc;
    *GetBlobDesc4BnInOp("hidden_ip_op_weight") = weight_blob_desc;
    if (GetBoolFromSpecialConf("has_bias_term")) {
      *GetBlobDesc4BnInOp("bias") =
          BlobDesc(Shape({1, hidden_size}), data_type, false);
      *GetBlobDesc4BnInOp("bias_multiplier") =
          BlobDesc(Shape({piece_size, 1}), data_type, false);
    }
  } else if (op_conf().recurrent_conf().rnn_type_case()
             == RecurrentOpConf::kBasicLstmCell) {
    TODO();
  } else {
    UNEXPECTED_RUN();
  }
}

std::string RecurrentOp::ibn2lbn(const std::string& input_bn) const {
  if (input_bn == "ht_1") { return obn2lbn("ht"); }
  return GetStringFromSpecialConf(input_bn);
}

REGISTER_OP(OperatorConf::kRecurrentConf, RecurrentOp);

}  // namespace oneflow
