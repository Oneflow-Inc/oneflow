#include "oneflow/core/operator/convolution_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

#ifdef WITH_CUDNN
namespace {

void InitCudnnTensorDesc(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ConvolutionOpConf& conv_conf, cudnnTensorDescriptor_t* in_desc,
    cudnnTensorDescriptor_t* out_desc, cudnnFilterDescriptor_t* filter_desc,
    cudnnConvolutionDescriptor_t* conv_desc) {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");

  CudaCheck(cudnnCreateTensorDescriptor(in_desc));
  CudaCheck(cudnnCreateTensorDescriptor(out_desc));
  CudaCheck(cudnnCreateFilterDescriptor(filter_desc));
  CudaCheck(cudnnCreateConvolutionDescriptor(conv_desc));

  cudnnDataType_t cudnn_data_type;
  switch (in_blob_desc->data_type()) {
    case kFloat: cudnn_data_type = CUDNN_DATA_FLOAT; break;
    case kDouble: cudnn_data_type = CUDNN_DATA_DOUBLE; break;
    default: UNEXPECTED_RUN();
  }

  CudaCheck(cudnnSetTensor4dDescriptor(
      *in_desc, CUDNN_TENSOR_NCHW, cudnn_data_type, in_blob_desc->shape().At(0),
      in_blob_desc->shape().At(1), in_blob_desc->shape().At(2),
      in_blob_desc->shape().At(3)));
  CudaCheck(cudnnSetTensor4dDescriptor(
      *out_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
      out_blob_desc->shape().At(0), out_blob_desc->shape().At(1),
      out_blob_desc->shape().At(2), out_blob_desc->shape().At(3)));
  CudaCheck(cudnnSetFilter4dDescriptor(
      *filter_desc, cudnn_data_type, CUDNN_TENSOR_NCHW,
      out_blob_desc->shape().At(1), in_blob_desc->shape().At(1),
      conv_conf.kernel_h(), conv_conf.kernel_w()));
  CudaCheck(cudnnSetConvolution2dDescriptor(
      *conv_desc, conv_conf.pad_h(), conv_conf.pad_w(), conv_conf.stride_h(),
      conv_conf.stride_w(), 1, 1, CUDNN_CROSS_CORRELATION, cudnn_data_type));
}

void DestroyCudnnTensorDesc(cudnnTensorDescriptor_t* in_desc,
                            cudnnTensorDescriptor_t* out_desc,
                            cudnnFilterDescriptor_t* filter_desc,
                            cudnnConvolutionDescriptor_t* conv_desc) {
  CudaCheck(cudnnDestroyConvolutionDescriptor(*conv_desc));
  CudaCheck(cudnnDestroyFilterDescriptor(*filter_desc));
  CudaCheck(cudnnDestroyTensorDescriptor(*out_desc));
  CudaCheck(cudnnDestroyTensorDescriptor(*in_desc));
}

void InferCudnnConvAlgo(const cudnnHandle_t* cudnn_handle,
                        cudnnTensorDescriptor_t* in_desc,
                        cudnnTensorDescriptor_t* out_desc,
                        cudnnFilterDescriptor_t* filter_desc,
                        cudnnConvolutionDescriptor_t* conv_desc,
                        cudnnConvolutionFwdAlgo_t* cudnn_fwd_algo,
                        cudnnConvolutionBwdFilterAlgo_t* cudnn_bwd_filter_algo,
                        cudnnConvolutionBwdDataAlgo_t* cudnn_bwd_data_algo) {
  CudaCheck(cudnnGetConvolutionForwardAlgorithm(
      *cudnn_handle, *in_desc, *filter_desc, *conv_desc, *out_desc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, cudnn_fwd_algo));
  CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm(
      *cudnn_handle, *in_desc, *out_desc, *conv_desc, *filter_desc,
      CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, cudnn_bwd_filter_algo));
  CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm(
      *cudnn_handle, *filter_desc, *out_desc, *conv_desc, *in_desc,
      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, cudnn_bwd_data_algo));
}

size_t ComputeCudnnConvWorkspaceSize(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ConvolutionOpConf& conv_conf) {
  CudaStreamHandle cuda_handle;

  cudnnTensorDescriptor_t in_desc;
  cudnnTensorDescriptor_t out_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(0);
  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(0);

  InitCudnnTensorDesc(GetBlobDesc4BnInOp, conv_conf, &in_desc, &out_desc,
                      &filter_desc, &conv_desc);
  InferCudnnConvAlgo(cuda_handle.cudnn_handle(), &in_desc, &out_desc,
                     &filter_desc, &conv_desc, &cudnn_fwd_algo,
                     &cudnn_bwd_filter_algo, &cudnn_bwd_data_algo);

  size_t cudnn_fwd_workspace_sizes = 0;
  size_t cudnn_bwd_filter_workspace_sizes = 0;
  size_t cudnn_bwd_data_workspace_sizes = 0;
  size_t cudnn_workspace_sizes = 0;

  // get workspace sizes of algorithm
  CudaCheck(cudnnGetConvolutionForwardWorkspaceSize(
      *cuda_handle.cudnn_handle(), in_desc, filter_desc, conv_desc, out_desc,
      cudnn_fwd_algo, &cudnn_fwd_workspace_sizes));
  CudaCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      *cuda_handle.cudnn_handle(), in_desc, out_desc, conv_desc, filter_desc,
      cudnn_bwd_filter_algo, &cudnn_bwd_filter_workspace_sizes));
  CudaCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(
      *cuda_handle.cudnn_handle(), filter_desc, out_desc, conv_desc, in_desc,
      cudnn_bwd_data_algo, &cudnn_bwd_data_workspace_sizes));

  cudnn_workspace_sizes = std::max(std::initializer_list<size_t>(
      {cudnn_fwd_workspace_sizes, cudnn_bwd_filter_workspace_sizes,
       cudnn_bwd_data_workspace_sizes}));

  DestroyCudnnTensorDesc(&in_desc, &out_desc, &filter_desc, &conv_desc);

  return cudnn_workspace_sizes;
}

void SetCudnnConfInConvKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ConvolutionOpConf& conv_conf,
    ConvolutionKernelConf* conv_kernel_conf) {
  CudaStreamHandle cuda_handle;

  cudnnTensorDescriptor_t in_desc;
  cudnnTensorDescriptor_t out_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(0);
  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(0);

  InitCudnnTensorDesc(GetBlobDesc4BnInOp, conv_conf, &in_desc, &out_desc,
                      &filter_desc, &conv_desc);
  InferCudnnConvAlgo(cuda_handle.cudnn_handle(), &in_desc, &out_desc,
                     &filter_desc, &conv_desc, &cudnn_fwd_algo,
                     &cudnn_bwd_filter_algo, &cudnn_bwd_data_algo);

  conv_kernel_conf->set_cudnn_fwd_algo(cudnn_fwd_algo);
  conv_kernel_conf->set_cudnn_bwd_filter_algo(cudnn_bwd_filter_algo);
  conv_kernel_conf->set_cudnn_bwd_data_algo(cudnn_bwd_data_algo);

  DestroyCudnnTensorDesc(&in_desc, &out_desc, &filter_desc, &conv_desc);
}

}  // namespace
#endif  // WITH_CUDNN

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

    if (!conf.use_cudnn()) {
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
  if (conf.use_cudnn()) {
    BlobDesc* cudnn_workspace_blob_desc = GetBlobDesc4BnInOp("cudnn_workspace");
    cudnn_workspace_blob_desc->mut_shape() = Shape({static_cast<int64_t>(
        ComputeCudnnConvWorkspaceSize(GetBlobDesc4BnInOp, conf))});
    cudnn_workspace_blob_desc->set_data_type(
        JobDesc::Singleton()->DefaultDataType());
    cudnn_workspace_blob_desc->set_has_data_id(false);
  }
#endif  // WITH_CUDNN

  if (!conf.use_cudnn()) {
    // col_buf
    BlobDesc* col_buf_blob_desc = GetBlobDesc4BnInOp("col_buf");
    CHECK(col_buf_blob_desc);
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
  if (op_conf().convolution_conf().use_cudnn()) {
    SetCudnnConfInConvKernelConf(GetBlobDesc4BnInOp,
                                 op_conf().convolution_conf(),
                                 kernel_conf->mutable_convolution_conf());
  }
#endif  // WITH_CUDNN
}

REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);

}  // namespace oneflow
