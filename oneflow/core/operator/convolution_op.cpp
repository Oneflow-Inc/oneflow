#include "oneflow/core/operator/convolution_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

#ifdef WITH_CUDNN
CudnnConvolutionOpUtil::CudnnConvolutionOpUtil() {
  CudaCheck(cudaStreamCreate(&cuda_stream_));
  CudaCheck(cudnnCreate(&cudnn_handle_));
  CudaCheck(cudnnSetStream(cudnn_handle_, cuda_stream_));

  CudaCheck(cudnnCreateTensorDescriptor(&in_desc_));
  CudaCheck(cudnnCreateTensorDescriptor(&out_desc_));
  CudaCheck(cudnnCreateFilterDescriptor(&filter_desc_));
  CudaCheck(cudnnCreateConvolutionDescriptor(&conv_desc_));
}

CudnnConvolutionOpUtil::~CudnnConvolutionOpUtil() {
  CudaCheck(cudnnDestroyTensorDescriptor(in_desc_));
  CudaCheck(cudnnDestroyTensorDescriptor(out_desc_));
  CudaCheck(cudnnDestroyConvolutionDescriptor(conv_desc_));
  CudaCheck(cudnnDestroyFilterDescriptor(filter_desc_));

  CudaCheck(cudaStreamDestroy(cuda_stream_));
  CudaCheck(cudnnDestroy(cudnn_handle_));
}

void CudnnConvolutionOpUtil::InitTensorDesc(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ConvolutionOpConf& conv_conf) {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");

  cudnnDataType_t cudnn_data_type;
  switch (JobDesc::Singleton()->DefaultDataType()) {
    case kFloat: cudnn_data_type = CUDNN_DATA_FLOAT; break;
    case kDouble: cudnn_data_type = CUDNN_DATA_DOUBLE; break;
    default: UNEXPECTED_RUN();
  }

  CudaCheck(cudnnSetTensor4dDescriptor(
      in_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type, in_blob_desc->shape().At(0),
      in_blob_desc->shape().At(1), in_blob_desc->shape().At(2),
      in_blob_desc->shape().At(3)));
  CudaCheck(cudnnSetTensor4dDescriptor(
      out_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type,
      out_blob_desc->shape().At(0), out_blob_desc->shape().At(1),
      out_blob_desc->shape().At(2), out_blob_desc->shape().At(3)));
  CudaCheck(cudnnSetFilter4dDescriptor(
      filter_desc_, cudnn_data_type, CUDNN_TENSOR_NCHW,
      out_blob_desc->shape().At(1), in_blob_desc->shape().At(1),
      conv_conf.kernel_h(), conv_conf.kernel_w()));
  CudaCheck(cudnnSetConvolution2dDescriptor(
      conv_desc_, conv_conf.pad_h(), conv_conf.pad_w(), conv_conf.stride_h(),
      conv_conf.stride_w(), 1, 1, CUDNN_CROSS_CORRELATION, cudnn_data_type));
}

cudnnConvolutionFwdAlgo_t CudnnConvolutionOpUtil::InferFwdAlgo() {
  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo =
      static_cast<cudnnConvolutionFwdAlgo_t>(0);

  CudaCheck(cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle_, in_desc_, filter_desc_, conv_desc_, out_desc_,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &cudnn_fwd_algo));

  return cudnn_fwd_algo;
}

cudnnConvolutionBwdFilterAlgo_t CudnnConvolutionOpUtil::InferBwdFilterAlgo() {
  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo =
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);

  CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle_, in_desc_, out_desc_, conv_desc_, filter_desc_,
      CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &cudnn_bwd_filter_algo));

  return cudnn_bwd_filter_algo;
}

cudnnConvolutionBwdDataAlgo_t CudnnConvolutionOpUtil::InferBwdDataAlgo() {
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo =
      static_cast<cudnnConvolutionBwdDataAlgo_t>(0);

  CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle_, filter_desc_, out_desc_, conv_desc_, in_desc_,
      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &cudnn_bwd_data_algo));

  return cudnn_bwd_data_algo;
}

size_t CudnnConvolutionOpUtil::InferWorkspaceSize(
    cudnnConvolutionFwdAlgo_t cudnn_fwd_algo,
    cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo,
    cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo) {
  size_t cudnn_fwd_workspace_sizes = 0;
  size_t cudnn_bwd_filter_workspace_sizes = 0;
  size_t cudnn_bwd_data_workspace_sizes = 0;
  size_t cudnn_workspace_sizes = 0;

  // get workspace sizes of algorithm
  CudaCheck(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn_handle_, in_desc_, filter_desc_, conv_desc_, out_desc_,
      cudnn_fwd_algo, &cudnn_fwd_workspace_sizes));
  CudaCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_handle_, in_desc_, out_desc_, conv_desc_, filter_desc_,
      cudnn_bwd_filter_algo, &cudnn_bwd_filter_workspace_sizes));
  CudaCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_handle_, filter_desc_, out_desc_, conv_desc_, in_desc_,
      cudnn_bwd_data_algo, &cudnn_bwd_data_workspace_sizes));

  cudnn_workspace_sizes = std::max(cudnn_bwd_filter_workspace_sizes,
                                   cudnn_bwd_data_workspace_sizes);
  cudnn_workspace_sizes =
      std::max(cudnn_workspace_sizes, cudnn_fwd_workspace_sizes);

  return cudnn_workspace_sizes;
}
#endif  // WITH_CUDNN

void ConvolutionOp::InitFromOpConf() {
  CHECK(op_conf().has_convolution_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");

  EnrollModelBn("weight");
  if (op_conf().convolution_conf().has_bias_term()) {
    EnrollModelBn("bias");
    if (!op_conf().convolution_conf().with_cudnn()) {
      EnrollModelTmpBn("bias_multiplier");
    }
  }

  if (op_conf().convolution_conf().with_cudnn()) {
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
  out_blob_desc->set_has_data_id(in_blob_desc->has_data_id());

  // weight
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() = Shape({c_o, c_i * kernel});
  weight_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
  weight_blob_desc->set_has_data_id(false);

  if (conf.has_bias_term()) {
    // bias
    BlobDesc* bias_blob_desc = GetBlobDesc4BnInOp("bias");
    bias_blob_desc->mut_shape() = Shape({c_o});
    bias_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
    bias_blob_desc->set_has_data_id(false);

    if (!conf.with_cudnn()) {
      // bias multiplier
      BlobDesc* bias_multiplier_blob_desc =
          GetBlobDesc4BnInOp("bias_multiplier");
      bias_multiplier_blob_desc->mut_shape() = Shape({output_size});
      bias_multiplier_blob_desc->set_data_type(
          JobDesc::Singleton()->DefaultDataType());
      bias_multiplier_blob_desc->set_has_data_id(false);
    }
  }

#ifdef WITH_CUDNN
  if (conf.with_cudnn()) {
    CudnnConvolutionOpUtil cudnn_conv_util;

    cudnn_conv_util.InitTensorDesc(GetBlobDesc4BnInOp, conf);
    cudnnConvolutionFwdAlgo_t cudnn_fwd_algo = cudnn_conv_util.InferFwdAlgo();
    cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo =
        cudnn_conv_util.InferBwdFilterAlgo();
    cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo =
        cudnn_conv_util.InferBwdDataAlgo();

    size_t cudnn_workspace_size = cudnn_conv_util.InferWorkspaceSize(
        cudnn_fwd_algo, cudnn_bwd_filter_algo, cudnn_bwd_data_algo);

    BlobDesc* cudnn_workspace_blob_desc = GetBlobDesc4BnInOp("cudnn_workspace");
    cudnn_workspace_blob_desc->mut_shape() =
        Shape({static_cast<int64_t>(cudnn_workspace_size)});
    cudnn_workspace_blob_desc->set_data_type(
        JobDesc::Singleton()->DefaultDataType());
    cudnn_workspace_blob_desc->set_has_data_id(false);
  }
#endif  // WITH_CUDNN

  if (!conf.with_cudnn()) {
    // col_buf
    BlobDesc* col_buf_blob_desc = GetBlobDesc4BnInOp("col_buf");
    col_buf_blob_desc->mut_shape() =
        Shape({data_num, output_size, c_i * kernel});
    col_buf_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
    col_buf_blob_desc->set_has_data_id(false);
  }
}

void ConvolutionOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
#ifdef WITH_CUDNN
  if (op_conf().convolution_conf().with_cudnn()) {
    CudnnConvolutionOpUtil cudnn_conv_util;

    cudnn_conv_util.InitTensorDesc(GetBlobDesc4BnInOp,
                                   op_conf().convolution_conf());
    cudnnConvolutionFwdAlgo_t cudnn_fwd_algo = cudnn_conv_util.InferFwdAlgo();
    cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo =
        cudnn_conv_util.InferBwdFilterAlgo();
    cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo =
        cudnn_conv_util.InferBwdDataAlgo();

    kernel_conf->mutable_convolution_conf()->set_cudnn_fwd_algo(
        static_cast<ConvolutionKernelConf::cudnnConvolutionFwdAlgo_t>(
            cudnn_fwd_algo));
    kernel_conf->mutable_convolution_conf()->set_cudnn_bwd_filter_algo(
        static_cast<ConvolutionKernelConf::cudnnConvolutionBwdFilterAlgo_t>(
            cudnn_bwd_filter_algo));
    kernel_conf->mutable_convolution_conf()->set_cudnn_bwd_data_algo(
        static_cast<ConvolutionKernelConf::cudnnConvolutionBwdDataAlgo_t>(
            cudnn_bwd_data_algo));
  }
#endif  // WITH_CUDNN
}

REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);

}  // namespace oneflow
