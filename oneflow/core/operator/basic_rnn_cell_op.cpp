#include "oneflow/core/operator/basic_rnn_cell_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void BasicRnnCellOp::InitFromOpConf() {
  CHECK(op_conf().has_basic_rnn_cell_conf());

  EnrollInputBn("in");
  EnrollInputBn("last_time_out");
  if (!GetStringFromSpecialConf("hidden_state").empty()) {
    EnrollInputBn("hidden_state");
  }
  EnrollOutputBn("out");

  if (op_conf().basic_rnn_cell_conf().rnn_type_case()
      == BasicRnnCellOpConf::kRnnCell) {
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
  } else if (op_conf().basic_rnn_cell_conf().rnn_type_case()
             == BasicRnnCellOpConf::kLstmCell) {
    TODO();
  } else {
    UNEXPECTED_RUN();
  }
}

const PbMessage& BasicRnnCellOp::GetSpecialConf() const {
  return op_conf().basic_rnn_cell_conf();
}

void BasicRnnCellOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BasicRnnCellOpConf& conf = op_conf().basic_rnn_cell_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  auto data_type = JobDesc::Singleton()->DefaultDataType();
  CHECK_EQ(in_blob_desc->data_type(), data_type);
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 2);
  if (!GetStringFromSpecialConf("hidden_state").empty()) {
    const BlobDesc* hid_blob_desc = GetBlobDesc4BnInOp("hidden_state");
    CHECK_EQ(hid_blob_desc->data_type(), data_type);
    CHECK_EQ(in_blob_desc->shape(), hid_blob_desc->shape());
  }
  int32_t embedding_size = in_blob_desc->shape().At(1);
  int32_t piece_size = in_blob_desc->shape().At(0);
  int32_t hidden_size = conf.hidden_size();
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(hidden_size, parallel_ctx->parallel_num());
    hidden_size = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({piece_size, hidden_size});
  out_blob_desc->set_data_type(data_type);
  out_blob_desc->set_has_data_id(in_blob_desc->has_data_id());

  if (op_conf().basic_rnn_cell_conf().rnn_type_case()
      == BasicRnnCellOpConf::kRnnCell) {
    *GetBlobDesc4BnInOp("in_ip_op_out") = *out_blob_desc;
    *GetBlobDesc4BnInOp("hidden_ip_op_out") = *out_blob_desc;
    *GetBlobDesc4BnInOp("plus_op_out") = *out_blob_desc;
    *GetBlobDesc4BnInOp("f_op_out") = *out_blob_desc;

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
  } else if (op_conf().basic_rnn_cell_conf().rnn_type_case()
             == BasicRnnCellOpConf::kLstmCell) {
    TODO();
  } else {
    UNEXPECTED_RUN();
  }
}

std::string BasicRnnCellOp::ibn2lbn(const std::string& input_bn) const {
  if (input_bn == "last_time_out") { return obn2lbn("out"); }
  return GetStringFromSpecialConf(input_bn);
}

REGISTER_OP(OperatorConf::kBasicRnnCellConf, BasicRnnCellOp);

}  // namespace oneflow
