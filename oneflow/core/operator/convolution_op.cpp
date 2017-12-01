#include "oneflow/core/operator/convolution_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

#ifdef USE_CUDNN
namespace {

void GetCudnnConvAlgo(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ConvolutionOpConf& conv_conf, cudaStream_t* cuda_stream,
    cudnnHandle_t* cudnn_handle, cudnnTensorDescriptor_t* in_desc,
    cudnnTensorDescriptor_t* out_desc, cudnnFilterDescriptor_t* filter_desc,
    cudnnConvolutionDescriptor_t* conv_desc,
    cudnnConvolutionFwdAlgo_t* cudnn_fwd_algo,
    cudnnConvolutionBwdFilterAlgo_t* cudnn_bwd_filter_algo,
    cudnnConvolutionBwdDataAlgo_t* cudnn_bwd_data_algo) {
  *cudnn_fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(0);
  *cudnn_bwd_filter_algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
  *cudnn_bwd_data_algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(0);

  CudaCheck(cudaStreamCreate(cuda_stream));
  CudaCheck(cudnnCreate(cudnn_handle));
  CudaCheck(cudnnSetStream(*cudnn_handle, *cuda_stream));

  CudaCheck(cudnnCreateTensorDescriptor(in_desc));
  CudaCheck(cudnnCreateTensorDescriptor(out_desc));
  CudaCheck(cudnnCreateFilterDescriptor(filter_desc));
  CudaCheck(cudnnCreateConvolutionDescriptor(conv_desc));

  cudnnDataType_t cudnn_data_type;
  switch (JobDesc::Singleton()->DefaultDataType()) {
    case kFloat: cudnn_data_type = CUDNN_DATA_FLOAT; break;
    case kDouble: cudnn_data_type = CUDNN_DATA_DOUBLE; break;
    default: UNEXPECTED_RUN();
  }

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");

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

void DestroyCudnnDescriptors(cudaStream_t* cuda_stream,
                             cudnnHandle_t* cudnn_handle,
                             cudnnTensorDescriptor_t* in_desc,
                             cudnnTensorDescriptor_t* out_desc,
                             cudnnConvolutionDescriptor_t* conv_desc,
                             cudnnFilterDescriptor_t* filter_desc) {
  CudaCheck(cudnnDestroyTensorDescriptor(*in_desc));
  CudaCheck(cudnnDestroyTensorDescriptor(*out_desc));
  CudaCheck(cudnnDestroyConvolutionDescriptor(*conv_desc));
  CudaCheck(cudnnDestroyFilterDescriptor(*filter_desc));
  CudaCheck(cudaStreamDestroy(*cuda_stream));
  CudaCheck(cudnnDestroy(*cudnn_handle));
}

}  // namespace
#endif  // USE_CUDNN

void ConvolutionOp::InitFromOpConf() {
  CHECK(op_conf().has_convolution_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");

  EnrollModelBn("weight");
  if (GetBoolFromSpecialConf("has_bias_term")) {
    EnrollModelBn("bias");
    if (!GetBoolFromSpecialConf("use_cudnn")) {
      EnrollModelTmpBn("bias_multiplier");
    }
  }

  if (GetBoolFromSpecialConf("use_cudnn")) {
    EnrollDataTmpBn("cudnn_fwd_workspace");
    EnrollDataTmpBn("cudnn_bwd_workspace");
  } else {
    EnrollDataTmpBn("col_buf");
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

    if (!GetBoolFromSpecialConf("use_cudnn")) {
      // bias multiplier
      BlobDesc* bias_multiplier_blob_desc =
          GetBlobDesc4BnInOp("bias_multiplier");
      bias_multiplier_blob_desc->mut_shape() = Shape({output_size});
      bias_multiplier_blob_desc->set_data_type(
          JobDesc::Singleton()->DefaultDataType());
      bias_multiplier_blob_desc->set_has_data_id(false);
    }
  }

  if (!GetBoolFromSpecialConf("use_cudnn")) {
    // col_buf
    BlobDesc* col_buf_blob_desc = GetBlobDesc4BnInOp("col_buf");
    col_buf_blob_desc->mut_shape() =
        Shape({data_num, output_size, c_i * kernel});
    col_buf_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
    col_buf_blob_desc->set_has_data_id(false);
  }

#ifdef USE_CUDNN
  if (conf.use_cudnn()) {
    std::unique_ptr<cudaStream_t> cuda_stream(new cudaStream_t);
    std::unique_ptr<cudnnHandle_t> cudnn_handle(new cudnnHandle_t);

    std::unique_ptr<cudnnTensorDescriptor_t> in_desc(
        new cudnnTensorDescriptor_t);
    std::unique_ptr<cudnnTensorDescriptor_t> out_desc(
        new cudnnTensorDescriptor_t);
    std::unique_ptr<cudnnFilterDescriptor_t> filter_desc(
        new cudnnFilterDescriptor_t);
    std::unique_ptr<cudnnConvolutionDescriptor_t> conv_desc(
        new cudnnConvolutionDescriptor_t);

    std::unique_ptr<cudnnConvolutionFwdAlgo_t> cudnn_fwd_algo(
        new cudnnConvolutionFwdAlgo_t);
    std::unique_ptr<cudnnConvolutionBwdFilterAlgo_t> cudnn_bwd_filter_algo(
        new cudnnConvolutionBwdFilterAlgo_t);
    std::unique_ptr<cudnnConvolutionBwdDataAlgo_t> cudnn_bwd_data_algo(
        new cudnnConvolutionBwdDataAlgo_t);

    GetCudnnConvAlgo(GetBlobDesc4BnInOp, op_conf().convolution_conf(),
                     cuda_stream.get(), cudnn_handle.get(), in_desc.get(),
                     out_desc.get(), filter_desc.get(), conv_desc.get(),
                     cudnn_fwd_algo.get(), cudnn_bwd_filter_algo.get(),
                     cudnn_bwd_data_algo.get());

    size_t cudnn_fwd_workspace_sizes = 0;
    size_t cudnn_bwd_filter_workspace_sizes = 0;
    size_t cudnn_bwd_data_workspace_sizes = 0;

    // get workspace sizes of algorithm
    CudaCheck(cudnnGetConvolutionForwardWorkspaceSize(
        *cudnn_handle, *in_desc, *filter_desc, *conv_desc, *out_desc,
        *cudnn_fwd_algo, &cudnn_fwd_workspace_sizes));
    CudaCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        *cudnn_handle, *in_desc, *out_desc, *conv_desc, *filter_desc,
        *cudnn_bwd_filter_algo, &cudnn_bwd_filter_workspace_sizes));
    CudaCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(
        *cudnn_handle, *filter_desc, *out_desc, *conv_desc, *in_desc,
        *cudnn_bwd_data_algo, &cudnn_bwd_data_workspace_sizes));

    BlobDesc* cudnn_fwd_workspace_blob_desc =
        GetBlobDesc4BnInOp("cudnn_fwd_workspace");
    cudnn_fwd_workspace_blob_desc->mut_shape() =
        Shape({static_cast<int64_t>(cudnn_fwd_workspace_sizes)});
    cudnn_fwd_workspace_blob_desc->set_data_type(
        JobDesc::Singleton()->DefaultDataType());
    cudnn_fwd_workspace_blob_desc->set_has_data_id(false);

    BlobDesc* cudnn_bwd_workspace_blob_desc =
        GetBlobDesc4BnInOp("cudnn_bwd_workspace");
    cudnn_bwd_workspace_blob_desc->mut_shape() =
        Shape({static_cast<int64_t>(std::max(cudnn_bwd_filter_workspace_sizes,
                                             cudnn_bwd_data_workspace_sizes))});
    cudnn_bwd_workspace_blob_desc->set_data_type(
        JobDesc::Singleton()->DefaultDataType());
    cudnn_bwd_workspace_blob_desc->set_has_data_id(false);

    DestroyCudnnDescriptors(cuda_stream.get(), cudnn_handle.get(),
                            in_desc.get(), out_desc.get(), conv_desc.get(),
                            filter_desc.get());
  }
#endif  // USE_CUDNN
}

void ConvolutionOp::GenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    bool is_forward, const ParallelContext* parallel_ctx,
    KernelConf* kernel_conf) const {
  Operator::GenKernelConf(GetBlobDesc4BnInOp, is_forward, parallel_ctx,
                          kernel_conf);
#ifdef USE_CUDNN
  if (GetBoolFromSpecialConf("use_cudnn")) {
    std::unique_ptr<cudaStream_t> cuda_stream(new cudaStream_t);
    std::unique_ptr<cudnnHandle_t> cudnn_handle(new cudnnHandle_t);

    std::unique_ptr<cudnnTensorDescriptor_t> in_desc(
        new cudnnTensorDescriptor_t);
    std::unique_ptr<cudnnTensorDescriptor_t> out_desc(
        new cudnnTensorDescriptor_t);
    std::unique_ptr<cudnnFilterDescriptor_t> filter_desc(
        new cudnnFilterDescriptor_t);
    std::unique_ptr<cudnnConvolutionDescriptor_t> conv_desc(
        new cudnnConvolutionDescriptor_t);

    std::unique_ptr<cudnnConvolutionFwdAlgo_t> cudnn_fwd_algo(
        new cudnnConvolutionFwdAlgo_t);
    std::unique_ptr<cudnnConvolutionBwdFilterAlgo_t> cudnn_bwd_filter_algo(
        new cudnnConvolutionBwdFilterAlgo_t);
    std::unique_ptr<cudnnConvolutionBwdDataAlgo_t> cudnn_bwd_data_algo(
        new cudnnConvolutionBwdDataAlgo_t);

    GetCudnnConvAlgo(GetBlobDesc4BnInOp, op_conf().convolution_conf(),
                     cuda_stream.get(), cudnn_handle.get(), in_desc.get(),
                     out_desc.get(), filter_desc.get(), conv_desc.get(),
                     cudnn_fwd_algo.get(), cudnn_bwd_filter_algo.get(),
                     cudnn_bwd_data_algo.get());

    auto mutable_conv_conf =
        kernel_conf->mutable_op_conf()->mutable_convolution_conf();
    mutable_conv_conf->set_cudnn_fwd_algo(
        static_cast<ConvolutionOpConf::cudnnConvolutionFwdAlgo_t>(
            *cudnn_fwd_algo));
    mutable_conv_conf->set_cudnn_bwd_filter_algo(
        static_cast<ConvolutionOpConf::cudnnConvolutionBwdFilterAlgo_t>(
            *cudnn_bwd_filter_algo));
    mutable_conv_conf->set_cudnn_bwd_data_algo(
        static_cast<ConvolutionOpConf::cudnnConvolutionBwdDataAlgo_t>(
            *cudnn_bwd_data_algo));

    DestroyCudnnDescriptors(cuda_stream.get(), cudnn_handle.get(),
                            in_desc.get(), out_desc.get(), conv_desc.get(),
                            filter_desc.get());
  }
#endif  // USE_CUDNN
}

REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);

}  // namespace oneflow
