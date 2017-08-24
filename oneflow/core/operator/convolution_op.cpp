#include "oneflow/core/operator/convolution_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ConvolutionOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_convolution_conf());
  mut_op_conf() = op_conf;

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

void ConvolutionOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  const ConvolutionOpConf& conf = op_conf().convolution_conf();
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  CHECK_EQ(in_blob_desc->data_type(), conf.in().data_type());
  CHECK_EQ(conf.in().data_type(), JobDesc::Singleton()->default_data_type());
  CHECK_EQ(conf.out().data_type(), JobDesc::Singleton()->default_data_type());
  int64_t batch_size = in_blob_desc->shape().At(0);
  int64_t c_i = in_blob_desc->shape().At(1);

  int32_t out_num = GetInt32FromSpecialConf("out_num");
  if (policy == kModelParallel) {
    BalancedSplitter splitter(out_num, parallel_num);
    out_num = splitter.At(parallel_id).size();
  }
  int64_t c_o = out_num;

  int64_t kernel_size = 1;
  int64_t output_size = 1;
  std::vector<int64_t> output_shape_vec = {batch_size, c_o};

  int64_t h_len =
      (in_blob_desc->shape().At(2) + 2 * conf.pad_h() - conf.kernel_size_h())
          / conf.stride_h()
      + 1;
  int64_t w_len =
      (in_blob_desc->shape().At(3) + 2 * conf.pad_w() - conf.kernel_size_w())
          / conf.stride_w()
      + 1;
  output_shape_vec.push_back(h_len);
  output_shape_vec.push_back(w_len);
  kernel_size *= conf.kernel_size_h();
  kernel_size *= conf.kernel_size_w();
  output_size *= h_len;
  output_size *= w_len;

  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  out_blob_desc->mut_shape() = Shape(output_shape_vec);
  out_blob_desc->set_data_type(conf.out().data_type());
  out_blob_desc->set_has_data_id(in_blob_desc->has_data_id());

  // col_buf
  BlobDesc* col_buf_blob_desc = GetBlobDesc4BnInOp("col_buf");
  col_buf_blob_desc->mut_shape() =
      Shape({batch_size, output_size, c_i * kernel_size});
  col_buf_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  col_buf_blob_desc->set_has_data_id(false);

  // weight
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() = Shape({c_o, c_i * kernel_size});
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
