#include "oneflow/core/operator/convolution_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#ifdef WITH_CUDNN
#include "oneflow/core/device/cudnn_support.h"
#endif  // WITH_CUDNN

namespace oneflow {

void ConvolutionOp::InitFromOpConf() {
  CHECK(op_conf().has_convolution_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");

  EnrollModelBn("weight");
  if (op_conf().convolution_conf().has_bias_term()) {
    EnrollModelBn("bias");
    if (!op_conf().convolution_conf().use_cudnn()) {
      EnrollModelTmpBn("bias_multiplier");
    }
  }

  if (op_conf().convolution_conf().use_cudnn()) {
    EnrollDataTmpBn("cudnn_workspace");
  } else {
    EnrollDataTmpBn("col_buf");
  }
}

const PbMessage& ConvolutionOp::GetSpecialConf() const {
  return op_conf().convolution_conf();
}

void ConvolutionOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ConvolutionOpConf& conf = op_conf().convolution_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
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
  out_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  // weight
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() =
      Shape({c_o, c_i, conf.kernel_h(), conf.kernel_w()});
  weight_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
  weight_blob_desc->set_has_data_id_field(false);

  if (conf.has_bias_term()) {
    // bias
    BlobDesc* bias_blob_desc = GetBlobDesc4BnInOp("bias");
    bias_blob_desc->mut_shape() = Shape({c_o});
    bias_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
    bias_blob_desc->set_has_data_id_field(false);

    if (!conf.use_cudnn()) {
      // bias multiplier
      BlobDesc* bias_multiplier_blob_desc =
          GetBlobDesc4BnInOp("bias_multiplier");
      bias_multiplier_blob_desc->mut_shape() = Shape({output_size});
      bias_multiplier_blob_desc->set_data_type(
          JobDesc::Singleton()->DefaultDataType());
      bias_multiplier_blob_desc->set_has_data_id_field(false);
    }
  }

#ifdef WITH_CUDNN
  if (conf.use_cudnn()) {
    CudaStreamHandle cuda_handle;
    CudnnConvolutionDesc conv_desc;
    conv_desc.InitFromBlobDescAndOpConf(GetBlobDesc4BnInOp("in"),
                                        GetBlobDesc4BnInOp("out"), conf);

    BlobDesc* cudnn_workspace_blob_desc = GetBlobDesc4BnInOp("cudnn_workspace");
    cudnn_workspace_blob_desc->mut_shape() = Shape({static_cast<int64_t>(
        conv_desc.InferWorkspaceSize(cuda_handle.cudnn_handle()))});
    cudnn_workspace_blob_desc->set_data_type(
        JobDesc::Singleton()->DefaultDataType());
    cudnn_workspace_blob_desc->set_has_data_id_field(false);
  }
#endif  // WITH_CUDNN

  if (!conf.use_cudnn()) {
    // col_buf
    BlobDesc* col_buf_blob_desc = GetBlobDesc4BnInOp("col_buf");
    CHECK(col_buf_blob_desc);
    col_buf_blob_desc->mut_shape() =
        Shape({data_num, output_size, c_i * kernel});
    col_buf_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
    col_buf_blob_desc->set_has_data_id_field(false);
  }
}

void ConvolutionOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
#ifdef WITH_CUDNN
  if (op_conf().convolution_conf().use_cudnn()) {
    CudaStreamHandle cuda_handle;
    CudnnConvolutionDesc conv_desc;
    conv_desc.InitFromBlobDescAndOpConf(GetBlobDesc4BnInOp("in"),
                                        GetBlobDesc4BnInOp("out"),
                                        op_conf().convolution_conf());

    kernel_conf->mutable_convolution_conf()->set_cudnn_fwd_algo(
        conv_desc.InferFwdAlgo(cuda_handle.cudnn_handle()));
    kernel_conf->mutable_convolution_conf()->set_cudnn_bwd_filter_algo(
        conv_desc.InferBwdFilterAlgo(cuda_handle.cudnn_handle()));
    kernel_conf->mutable_convolution_conf()->set_cudnn_bwd_data_algo(
        conv_desc.InferBwdDataAlgo(cuda_handle.cudnn_handle()));
  }
#endif  // WITH_CUDNN
}

REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);

}  // namespace oneflow
