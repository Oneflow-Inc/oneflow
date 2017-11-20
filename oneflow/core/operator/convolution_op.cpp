#include "oneflow/core/operator/convolution_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ConvolutionOp::InitFromOpConf() {
  CHECK(op_conf().has_convolution_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("col_buf");

  EnrollModelBn("weight");
  if (GetBoolFromSpecialConf("has_bias_term")) {
    EnrollModelBn("bias");
    EnrollModelTmpBn("bias_multiplier");
  }
}

const PbMessage& ConvolutionOp::GetSpecialConf() const {
  return op_conf().convolution_conf();
}

void ConvolutionOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  const ConvolutionOpConf& conf = op_conf().convolution_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ(in_blob_desc->data_type(),
           JobDesc::Singleton()->default_data_type());
  int64_t data_num = in_blob_desc->shape().At(0);
  int64_t c_i = in_blob_desc->shape().At(1);

  int32_t out_num = GetInt32FromSpecialConf("out_num");
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(out_num, parallel_ctx->parallel_num());
    out_num = splitter.At(parallel_ctx->parallel_id()).size();
  }
  int64_t c_o = out_num;

  int64_t h_len =
      (in_blob_desc->shape().At(2) + 2 * conf.pad_h() - conf.kernel_h())
          / conf.stride_h()
      + 1;
  int64_t w_len =
      (in_blob_desc->shape().At(3) + 2 * conf.pad_w() - conf.kernel_w())
          / conf.stride_w()
      + 1;
  int64_t output_size = h_len * w_len;
  int64_t kernel = conf.kernel_h() * conf.kernel_w();

  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  out_blob_desc->mut_shape() = Shape({data_num, c_o, h_len, w_len});
  out_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  out_blob_desc->set_has_data_id(in_blob_desc->has_data_id());

  // col_buf
  BlobDesc* col_buf_blob_desc = GetBlobDesc4BnInOp("col_buf");
  col_buf_blob_desc->mut_shape() = Shape({data_num, output_size, c_i * kernel});
  col_buf_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  col_buf_blob_desc->set_has_data_id(false);

  // weight
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() = Shape({c_o, c_i * kernel});
  weight_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  weight_blob_desc->set_has_data_id(false);

  if (conf.has_bias_term()) {
    // bias
    BlobDesc* bias_blob_desc = GetBlobDesc4BnInOp("bias");
    bias_blob_desc->mut_shape() = Shape({c_o});
    bias_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
    bias_blob_desc->set_has_data_id(false);

    // bias multiplier
    BlobDesc* bias_multiplier_blob_desc = GetBlobDesc4BnInOp("bias_multiplier");
    bias_multiplier_blob_desc->mut_shape() = Shape({output_size});
    bias_multiplier_blob_desc->set_data_type(
        JobDesc::Singleton()->default_data_type());
    bias_multiplier_blob_desc->set_has_data_id(false);
  }
}

REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);

}  // namespace oneflow
