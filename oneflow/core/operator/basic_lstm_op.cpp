#include "oneflow/core/operator/basic_lstm_op.h"

namespace oneflow {

const PbMessage& BasicLstmOp::GetCustomizedConf() const {
  return op_conf().basic_lstm_conf();
}

void BasicLstmOp::VirtualInitFromOpConf() {
  EnrollDataTmpBn("f_gate_out");
  EnrollDataTmpBn("i_gate_out");
  EnrollDataTmpBn("o_gate_out");
  EnrollDataTmpBn("c_gate_out");

  EnrollModelBn("i2h_f_weight");
  EnrollModelBn("h2h_f_weight");
  EnrollModelBn("i2h_i_weight");
  EnrollModelBn("h2h_i_weight");
  EnrollModelBn("i2h_o_weight");
  EnrollModelBn("h2h_o_weight");
  EnrollModelBn("i2h_c_weight");
  EnrollModelBn("h2h_c_weight");
}

void BasicLstmOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t hidden_size = GetBlobDesc4BnInOp("out")->shape().At(1);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t embedding_zie = in_blob_desc->shape().Count(1);
  int64_t data_num = in_blob_desc->shape().At(0);

  *GetBlobDesc4BnInOp("f_gate_out") = BlobDesc(
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(),
      false, true, in_blob_desc->max_col_num());
  *GetBlobDesc4BnInOp("i_gate_out") = BlobDesc(
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(),
      false, true, in_blob_desc->max_col_num());
  *GetBlobDesc4BnInOp("o_gate_out") = BlobDesc(
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(),
      false, true, in_blob_desc->max_col_num());
  *GetBlobDesc4BnInOp("c_gate_out") = BlobDesc(
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(),
      false, true, in_blob_desc->max_col_num());

  *GetBlobDesc4BnInOp("i2h_f_weight") =
      BlobDesc(Shape({hidden_size, embedding_zie}));
  *GetBlobDesc4BnInOp("h2h_f_weight") =
      BlobDesc(Shape({hidden_size, embedding_zie}));
  *GetBlobDesc4BnInOp("i2h_i_weight") =
      BlobDesc(Shape({hidden_size, embedding_zie}));
  *GetBlobDesc4BnInOp("h2h_i_weight") =
      BlobDesc(Shape({hidden_size, embedding_zie}));
  *GetBlobDesc4BnInOp("i2h_o_weight") =
      BlobDesc(Shape({hidden_size, embedding_zie}));
  *GetBlobDesc4BnInOp("h2h_o_weight") =
      BlobDesc(Shape({hidden_size, embedding_zie}));
  *GetBlobDesc4BnInOp("i2h_c_weight") =
      BlobDesc(Shape({hidden_size, embedding_zie}));
  *GetBlobDesc4BnInOp("h2h_c_weight") =
      BlobDesc(Shape({hidden_size, embedding_zie}));
}
}  // namespace oneflow
