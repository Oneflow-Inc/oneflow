#include "oneflow/core/operator/recurrent_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void RecurrentOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollInputBn("rec_in");
  if (!GetStringFromSpecialConf("init_hidden").empty()) {
    CHECK(!GetBoolFromSpecialConf("has_init_hidden_initializer"));
    EnrollInputBn("h0");
  } else {
    EnrollModelBn("h0");
  }
  EnrollOutputBn("out");
  EnrollOutputBn("rec_out");
  VirtualInitFromOpConf();
}

int32_t RecurrentOp::MaxModelSplitNum() const {
  return GetInt32FromSpecialConf("hidden_size");
}

void RecurrentOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  DataType data_type = JobDesc::Singleton()->DefaultDataType();
  CHECK_EQ(in_blob_desc->data_type(), data_type);
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(in_blob_desc->has_col_num_field(), true);
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t hidden_size = GetInt32FromSpecialConf("hidden_size");
  Shape h0_shape = Shape({data_num, hidden_size});
  if (!GetStringFromSpecialConf("init_hidden").empty()) {
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

  VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

std::string RecurrentOp::ibn2lbn(const std::string& input_bn) const {
  if (input_bn == "rec_in") {
    return obn2lbn("rec_out");
  } else if (input_bn == "h0") {
    return GetStringFromSpecialConf("init_hidden");
  } else if (input_bn == "in") {
    return GetStringFromSpecialConf("in");
  } else {
    UNEXPECTED_RUN();
    return "";
  }
}

std::string RecurrentOp::obn2lbn(const std::string& output_bn) const {
  if (output_bn == "out") {
    return op_name() + "/" + GetStringFromSpecialConf("out");
  } else if (output_bn == "rec_out") {
    return op_name() + "/rec_" + GetStringFromSpecialConf("out");
  } else {
    UNEXPECTED_RUN();
    return "";
  }
}

}  // namespace oneflow
