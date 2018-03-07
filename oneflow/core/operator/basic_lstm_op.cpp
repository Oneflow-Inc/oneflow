#include "oneflow/core/operator/basic_lstm_op.h"

namespace oneflow {

const PbMessage& BasicLstmOp::GetCustomizedConf() const {
  return op_conf().basic_lstm_conf();
}

void BasicLstmOp::VirtualInitFromOpConf() {
  EnrollDataTmpBn("fgate_out");
  EnrollDataTmpBn("igate_out");
  EnrollDataTmpBn("ogate_out");
  EnrollDataTmpBn("cgate_out");

  EnrollModelBn("i2h_f_weight");
  EnrollModelBn("h2h_f_weight");
  EnrollModelBn("i2h_i_weight");
  EnrollModelBn("h2h_i_weight");
  EnrollModelBn("i2h_o_weight");
  EnrollModelBn("h2h_o_weight");
  EnrollModelBn("i2h_c_weight");
  EnrollModelBn("h2h_c_weight");
}

void BasicLstmOp::VirtualInferBobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t hidden_size = GetBlobDesc4BnInOp("out")->shape().At(1);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t embedding_size = in_blob_desc->shape().Count(1);
  int64_t data_num = in_blob_desc->shape().At(0);

  *GetBlobDesc4BnInOp("fgate_out") = BlobDesc(
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(),
      false, true, in_blob_desc->max_col_num());
  *GetBlobDesc4BnInOp("igate_out") = BlobDesc(
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(),
      false, true, in_blob_desc->max_col_num());
  *GetBlobDesc4BnInOp("ogate_out") = BlobDesc(
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(),
      false, true, in_blob_desc->max_col_num());
  *GetBlobDesc4BnInOp("cgate_out") = BlobDesc(
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(),
      false, true, in_blob_desc->max_col_num());
  *GetBlobDesc4BnInOp("i2h_f_weight") =
      BlobDesc(Shape({hidden_size, embedding_size}));
  *GetBlobDesc4BnInOp("h2h_f_weight") =
      BlobDesc(Shape({hidden_size, hidden_size}));
  *GetBlobDesc4BnInOp("i2h_i_weight") =
      BlobDesc(Shape({hidden_size, embedding_size}));
  *GetBlobDesc4BnInOp("h2h_i_weight") =
      BlobDesc(Shape({hidden_size, hidden_size}));
  *GetBlobDesc4BnInOp("i2h_o_weight") =
      BlobDesc(Shape({hidden_size, embedding_size}));
  *GetBlobDesc4BnInOp("h2h_o_weight") =
      BlobDesc(Shape({hidden_size, hidden_size}));
  *GetBlobDesc4BnInOp("i2h_c_weight") =
      BlobDesc(Shape({hidden_size, embedding_size}));
  *GetBlobDesc4BnInOp("h2h_c_weight") =
      BlobDesc(Shape({hidden_size, hidden_size}));
 }
}
REGISTER_OP(OperatorConf::kBasicLstmConf, BasicLstmOp);

}  // namespace oneflow
