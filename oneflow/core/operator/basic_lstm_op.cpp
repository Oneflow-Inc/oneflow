#include "oneflow/core/operator/basic_lstm_op.h"

namespace oneflow {

const PbMessage& BasicLstmOp::GetCustomizedConf() const {
  return op_conf().basic_lstm_conf();
}

void BasicLstmOp::VirtualInitFromOpConf() {
  EnrollDataTmpBn("f_gate_out");
  EnrollDataTmpBn("f_out");
  EnrollModelBn("i2h_f_weight");
  EnrollModelBn("h2h_f_weight");

  EnrollDataTmpBn("i_gate_out");
  EnrollDataTmpBn("i_out");
  EnrollModelBn("i2h_i_weight");
  EnrollModelBn("h2h_i_weight");

  EnrollDataTmpBn("c_state_out");
  EnrollDataTmpBn("c_out");
  EnrollModelBn("i2h_c_weight");
  EnrollModelBn("h2h_c_weight");

  EnrollDataTmpBn("o_gate_out");
  EnrollDataTmpBn("o_out");
  EnrollModelBn("i2h_o_weight");
  EnrollModelBn("h2h_o_weight");

  EnrollDataTmpBn("update_out");

  EnrollModelBn("bias_f");
  EnrollModelTmpBn("bias_f_multiplier");
  EnrollModelBn("bias_i");
  EnrollModelTmpBn("bias_i_multiplier");
  EnrollModelBn("bias_c");
  EnrollModelTmpBn("bias_c_multiplier");
  EnrollModelBn("bias_o");
  EnrollModelTmpBn("bias_o_multiplier");
}

void BasicLstmOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t hidden_size = GetBlobDesc4BnInOp("out")->shape().At(1);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t embedding_size = in_blob_desc->shape().Count(1);
  int64_t data_num = in_blob_desc->shape().At(0);

#define OF_INFER_LSTM_GATE_BLOBDESC(gate_name)                                 \
  *GetBlobDesc4BnInOp(#gate_name) = BlobDesc(                                  \
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(), \
      false, true, in_blob_desc->max_col_num())

#define OF_INFER_LSTM_MODEL_BLOBDESC(model_i2h_name, model_h2h_name) \
  *GetBlobDesc4BnInOp(#model_i2h_name) =                             \
      BlobDesc(Shape({hidden_size, embedding_size}));                \
  *GetBlobDesc4BnInOp(#model_h2h_name) =                             \
      BlobDesc(Shape({hidden_size, embedding_size}));

#define OF_INFER_LSTM_BIAS_BLOBDESC(bias_name, bias_mul_name)          \
  *GetBlobDesc4BnInOp(#bias_name) = BlobDesc(Shape({1, hidden_size})); \
  *GetBlobDesc4BnInOp(#bias_mul_name) = BlobDesc(Shape({data_num, 1}));

  OF_INFER_LSTM_GATE_BLOBDESC(f_gate_out);
  OF_INFER_LSTM_GATE_BLOBDESC(f_out);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_f_weight, h2h_f_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_f, bias_f_multiplier);

  OF_INFER_LSTM_GATE_BLOBDESC(i_gate_out);
  OF_INFER_LSTM_GATE_BLOBDESC(i_out);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_i_weight, h2h_i_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_i, bias_i_multiplier);

  OF_INFER_LSTM_GATE_BLOBDESC(c_state_out);
  OF_INFER_LSTM_GATE_BLOBDESC(c_out);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_c_weight, h2h_c_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_c, bias_c_multiplier);

  OF_INFER_LSTM_GATE_BLOBDESC(o_gate_out);
  OF_INFER_LSTM_GATE_BLOBDESC(o_out);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_o_weight, h2h_o_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_o, bias_o_multiplier);

  OF_INFER_LSTM_GATE_BLOBDESC(update);
#undef OF_INFER_LSTM_GATE_BLOBDESC
#undef OF_INFER_LSTM_MODEL_BLOBDESC
#undef OF_INFER_LSTM_BIAS_BLOBDESC
}

REGISTER_OP(OperatorConf::kBasicLstmConf, BasicLstmOp);

}  // namespace oneflow
