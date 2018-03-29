#include "oneflow/core/operator/basic_lstm_op.h"

namespace oneflow {

const PbMessage& BasicLstmOp::GetCustomizedConf() const {
  return op_conf().basic_lstm_conf();
}

void BasicLstmOp::VirtualInitFromOpConf() {
  EnrollDataTmpBn("state_data");
  EnrollDataTmpBn("candidate_out");
#define OF_INIT_LSTM_GATE_FROM_OP_CONF(out_name, i2h_weight, h2h_weight,    \
                                       data_diff, out_diff, bias, bias_mul) \
  EnrollDataTmpBn(#out_name);                                               \
  EnrollDataTmpBn(#data_diff);                                              \
  EnrollDataTmpBn(#out_diff);                                               \
  EnrollModelBn(#i2h_weight);                                               \
  EnrollModelBn(#h2h_weight);                                               \
  EnrollModelBn(#bias);                                                     \
  EnrollModelTmpBn(#bias_mul);

  OF_INIT_LSTM_GATE_FROM_OP_CONF(f_out, i2h_f_weight, h2h_f_weight, f_data_diff,
                                 f_out_diff, bias_f, bias_f_multiplier);
  OF_INIT_LSTM_GATE_FROM_OP_CONF(i_out, i2h_i_weight, h2h_i_weight, i_data_diff,
                                 i_out_diff, bias_i, bias_i_multiplier);
  OF_INIT_LSTM_GATE_FROM_OP_CONF(c_out, i2h_c_weight, h2h_c_weight, c_data_diff,
                                 c_out_diff, bias_c, bias_c_multiplier);
  OF_INIT_LSTM_GATE_FROM_OP_CONF(o_out, i2h_o_weight, h2h_o_weight, o_data_diff,
                                 o_out_diff, bias_i, bias_i_multiplier);
#undef OF_INIT_LSTM_GATE_FROM_OP_CONF
}

void BasicLstmOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t hidden_size = GetBlobDesc4BnInOp("out")->shape().At(1);
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
      BlobDesc(Shape({hidden_size, embedding_size}));

#define OF_INFER_LSTM_BIAS_BLOBDESC(bias_name, bias_mul_name)          \
  *GetBlobDesc4BnInOp(#bias_name) = BlobDesc(Shape({1, hidden_size})); \
  *GetBlobDesc4BnInOp(#bias_mul_name) = BlobDesc(Shape({data_num, 1}));

  OF_INFER_LSTM_GATE_BLOBDESC(state_data);
  OF_INFER_LSTM_GATE_BLOBDESC(candidate_out);

  OF_INFER_LSTM_GATE_BLOBDESC(f_out);
  OF_INFER_LSTM_GATE_BLOBDESC(f_data_diff);
  OF_INFER_LSTM_GATE_BLOBDESC(f_out_diff);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_f_weight, h2h_f_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_f, bias_f_multiplier);

  OF_INFER_LSTM_GATE_BLOBDESC(i_out);
  OF_INFER_LSTM_GATE_BLOBDESC(i_data_diff);
  OF_INFER_LSTM_GATE_BLOBDESC(i_out_diff);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_i_weight, h2h_i_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_i, bias_i_multiplier);

  OF_INFER_LSTM_GATE_BLOBDESC(c_out);
  OF_INFER_LSTM_GATE_BLOBDESC(c_data_diff);
  OF_INFER_LSTM_GATE_BLOBDESC(c_out_diff);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_c_weight, h2h_c_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_c, bias_c_multiplier);

  OF_INFER_LSTM_GATE_BLOBDESC(o_out);
  OF_INFER_LSTM_GATE_BLOBDESC(o_data_diff);
  OF_INFER_LSTM_GATE_BLOBDESC(o_out_diff);
  OF_INFER_LSTM_MODEL_BLOBDESC(i2h_o_weight, h2h_o_weight);
  OF_INFER_LSTM_BIAS_BLOBDESC(bias_o, bias_o_multiplier);

#undef OF_INFER_LSTM_GATE_BLOBDESC
#undef OF_INFER_LSTM_MODEL_BLOBDESC
#undef OF_INFER_LSTM_BIAS_BLOBDESC
}

REGISTER_OP(OperatorConf::kBasicLstmConf, BasicLstmOp);

}  // namespace oneflow
