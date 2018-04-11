#include "oneflow/core/operator/basic_lstm_op.h"

namespace oneflow {

const PbMessage& BasicLstmOp::GetSpecialConf() const {
  return op_conf().basic_lstm_conf();
}

void BasicLstmOp::VirtualInitFromOpConf() {
  InitCellFromOpConf();
#define OF_INIT_LSTM_GATE_FROM_OP_CONF(out_name, i2h_weight, h2h_weight, \
                                       data_diff, out_diff)              \
  EnrollDataTmpBn(#out_name);                                            \
  EnrollDataTmpBn(#data_diff);                                           \
  EnrollDataTmpBn(#out_diff);                                            \
  EnrollModelBn(#i2h_weight);                                            \
  EnrollModelBn(#h2h_weight);
  // for f,i.o gate
  OF_INIT_LSTM_GATE_FROM_OP_CONF(f_out, i2h_f_weight, h2h_f_weight, f_data_diff,
                                 f_out_diff);
  OF_INIT_LSTM_GATE_FROM_OP_CONF(i_out, i2h_i_weight, h2h_i_weight, i_data_diff,
                                 i_out_diff);
  OF_INIT_LSTM_GATE_FROM_OP_CONF(o_out, i2h_o_weight, h2h_o_weight, o_data_diff,
                                 o_out_diff);
  // for c_gate
  EnrollDataTmpBn("c_data");
  OF_INIT_LSTM_GATE_FROM_OP_CONF(c_out, i2h_c_weight, h2h_c_weight, c_data_diff,
                                 c_out_diff);

  // candidate_data = rec_cell_out
  EnrollDataTmpBn("candidate_data");
  // candidate_out = tanh(rec_cell_out)
  EnrollDataTmpBn("candidate_out");

  if (GetBoolFromSpecialConf("use_forget_bias")) { EnrollModelBn("bias_f"); }
  EnrollModelBn("bias_f");
  EnrollModelBn("bias_i");
  EnrollModelBn("bais_c");
  EnrollModelBn("bias_o");
  EnrollModelTmpBn("bias_multiplier");
#undef OF_INIT_LSTM_GATE_FROM_OP_CONF
}

void BasicLstmOp::InitCellFromOpConf() {
  EnrollInputBn("rec_cell_in");
  if (!GetStringFromSpecialConf("init_cell").empty()) {
    CHECK(!GetBoolFromSpecialConf("has_init_cell_initializer"));
    EnrollInputBn("c0");
  } else {
    EnrollModelTmpBn("c0");
  }
  EnrollOutputBn("rec_cell_out");
}

void BasicLstmOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int64_t hidden_size = GetBlobDesc4BnInOp("out")->shape().At(1);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t embedding_size = in_blob_desc->shape().Count(1);
  int64_t data_num = in_blob_desc->shape().At(0);

#define OF_INFER_LSTM_GATE_BLOBDESC(model_name)                                \
  *GetBlobDesc4BnInOp(#model_name) = BlobDesc(                                 \
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(), \
      false, true, in_blob_desc->shape().At(0));

#define OF_INFER_LSTM_MODEL_BLOBDESC(model_i2h_name, model_h2h_name) \
  *GetBlobDesc4BnInOp(#model_i2h_name) =                             \
      BlobDesc(Shape({hidden_size, embedding_size}));                \
  *GetBlobDesc4BnInOp(#model_h2h_name) =                             \
      BlobDesc(Shape({hidden_size, hidden_size}));

#define OF_INFER_LSTM_BIAS_BLOBDESC(bias_name) \
  *GetBlobDesc4BnInOp(#bias_name) = BlobDesc(Shape({1, hidden_size}));

  OF_INFER_LSTM_GATE_BLOBDESC(gate_tmp_data);
  OF_INFER_LSTM_GATE_BLOBDESC(candidate_out);

  OF_INFER_LSTM_GATE_BLOBDESC(candidate_data);
  OF_INFER_LSTM_GATE_BLOBDESC(c_data);

  OF_INFER_LSTM_GATE_BLOBDESC(f_out);
  OF_INFER_LSTM_GATE_BLOBDESC(f_data_diff);
  OF_INFER_LSTM_GATE_BLOBDESC(f_out_diff);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_f_weight, h2h_f_weight);
  if (GetBoolFromSpecialConf("use_forget_bias")) {
    OF_INFER_LSTM_BIAS_BLOBDESC(bias_f);
  }
  OF_INFER_LSTM_GATE_BLOBDESC(i_out);
  OF_INFER_LSTM_GATE_BLOBDESC(i_data_diff);
  OF_INFER_LSTM_GATE_BLOBDESC(i_out_diff);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_i_weight, h2h_i_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_i);

  OF_INFER_LSTM_GATE_BLOBDESC(c_out);
  OF_INFER_LSTM_GATE_BLOBDESC(c_data_diff);
  OF_INFER_LSTM_GATE_BLOBDESC(c_out_diff);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_c_weight, h2h_c_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_c);

  OF_INFER_LSTM_GATE_BLOBDESC(o_out);
  OF_INFER_LSTM_GATE_BLOBDESC(o_data_diff);
  OF_INFER_LSTM_GATE_BLOBDESC(o_out_diff);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_o_weight, h2h_o_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_o);

  *GetBlobDesc4BnInOp("bias_multiplier") = BlobDesc(Shape({data_num, 1}));

#undef OF_INFER_LSTM_GATE_BLOBDESC
#undef OF_INFER_LSTM_MODEL_BLOBDESC
#undef OF_INFER_LSTM_BIAS_BLOBDESC
}

std::string BasicLstmOp::VirtualIbn2Lbn(const std::string& input_bn) const {
  if (input_bn == "rec_cell_in") {
    return VirtualObn2Lbn("rec_cell_out");
  } else if (input_bn == "c0") {
    return GetStringFromSpecialConf("init_cell");
  } else {
    UNEXPECTED_RUN();
    return "";
  }
}

std::string BasicLstmOp::VirtualObn2Lbn(const std::string& output_bn) const {
  if (output_bn == "rec_cell_out") {
    return op_name() + "/rec_cell" + GetStringFromSpecialConf("out");
  } else {
    UNEXPECTED_RUN();
    return "";
  }
}
REGISTER_OP(OperatorConf::kBasicLstmConf, BasicLstmOp);

}  // namespace oneflow
