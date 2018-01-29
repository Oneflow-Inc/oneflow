#include "oneflow/core/operator/conv2d_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#ifdef WITH_CUDNN
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cudnn_util.h"
#endif  // WITH_CUDNN

namespace oneflow {

#ifdef WITH_CUDNN
namespace {

cudnnConvolutionFwdAlgo_t InferCudnnConvFwdAlgo(
    const cudnnHandle_t& cudnn_handle, CudnnTensorDesc* in_desc,
    CudnnTensorDesc* out_desc, CudnnFilterDesc* filter_desc,
    CudnnConvolutionDesc* conv_desc) {
  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo;

  CudaCheck(cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle, in_desc->Get(), filter_desc->Get(), conv_desc->Get(),
      out_desc->Get(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,
      &cudnn_fwd_algo));

  return cudnn_fwd_algo;
}

cudnnConvolutionBwdFilterAlgo_t InferCudnnConvBwdFilterAlgo(
    const cudnnHandle_t& cudnn_handle, CudnnTensorDesc* in_desc,
    CudnnTensorDesc* out_desc, CudnnFilterDesc* filter_desc,
    CudnnConvolutionDesc* conv_desc) {
  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo;

  CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle, in_desc->Get(), out_desc->Get(), conv_desc->Get(),
      filter_desc->Get(), CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
      &cudnn_bwd_filter_algo));

  return cudnn_bwd_filter_algo;
}

cudnnConvolutionBwdDataAlgo_t InferCudnnConvBwdDataAlgo(
    const cudnnHandle_t& cudnn_handle, CudnnTensorDesc* in_desc,
    CudnnTensorDesc* out_desc, CudnnFilterDesc* filter_desc,
    CudnnConvolutionDesc* conv_desc) {
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo;

  CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle, filter_desc->Get(), out_desc->Get(), conv_desc->Get(),
      in_desc->Get(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
      &cudnn_bwd_data_algo));

  return cudnn_bwd_data_algo;
}

void SetCudnnConvAlgoForKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const Conv2dOpConf& conv_op_conf, Conv2dKernelConf* conv_kernel_conf) {
  CudaStreamHandle cuda_handle;

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* filter_blob_desc = GetBlobDesc4BnInOp("filter");

  DataType data_type = in_blob_desc->data_type();

  CudnnTensorDesc in_desc(data_type, in_blob_desc->shape());
  CudnnTensorDesc out_desc(data_type, out_blob_desc->shape());
  CudnnFilterDesc filter_desc(data_type, filter_blob_desc->shape());
  CudnnConvolutionDesc conv_desc(data_type, conv_op_conf);

  conv_kernel_conf->set_cudnn_fwd_algo(
      InferCudnnConvFwdAlgo(*cuda_handle.cudnn_handle(), &in_desc, &out_desc,
                            &filter_desc, &conv_desc));
  conv_kernel_conf->set_cudnn_bwd_filter_algo(
      InferCudnnConvBwdFilterAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                  &out_desc, &filter_desc, &conv_desc));
  conv_kernel_conf->set_cudnn_bwd_data_algo(
      InferCudnnConvBwdDataAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                &out_desc, &filter_desc, &conv_desc));
}

size_t InferCudnnWorkspaceSize(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const Conv2dOpConf& conv_op_conf) {
  size_t fwd_workspace_size = 0;
  size_t bwd_filter_workspace_size = 0;
  size_t bwd_data_workspace_size = 0;

  CudaStreamHandle cuda_handle;

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* filter_blob_desc = GetBlobDesc4BnInOp("filter");

  DataType data_type = in_blob_desc->data_type();

  CudnnTensorDesc in_desc(data_type, in_blob_desc->shape());
  CudnnTensorDesc out_desc(data_type, out_blob_desc->shape());
  CudnnFilterDesc filter_desc(data_type, filter_blob_desc->shape());
  CudnnConvolutionDesc conv_desc(data_type, conv_op_conf);

  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo =
      InferCudnnConvFwdAlgo(*cuda_handle.cudnn_handle(), &in_desc, &out_desc,
                            &filter_desc, &conv_desc);
  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo =
      InferCudnnConvBwdFilterAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                  &out_desc, &filter_desc, &conv_desc);
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo =
      InferCudnnConvBwdDataAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                &out_desc, &filter_desc, &conv_desc);

  CudaCheck(cudnnGetConvolutionForwardWorkspaceSize(
      *cuda_handle.cudnn_handle(), in_desc.Get(), filter_desc.Get(),
      conv_desc.Get(), out_desc.Get(), cudnn_fwd_algo, &fwd_workspace_size));
  CudaCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      *cuda_handle.cudnn_handle(), in_desc.Get(), out_desc.Get(),
      conv_desc.Get(), filter_desc.Get(), cudnn_bwd_filter_algo,
      &bwd_filter_workspace_size));
  CudaCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(
      *cuda_handle.cudnn_handle(), filter_desc.Get(), out_desc.Get(),
      conv_desc.Get(), in_desc.Get(), cudnn_bwd_data_algo,
      &bwd_data_workspace_size));

  return std::max(
      {fwd_workspace_size, bwd_filter_workspace_size, bwd_data_workspace_size});
}

}  // namespace
#endif  // WITH_CUDNN

void Conv2dOp::InitFromOpConf() {
  CHECK(op_conf().has_conv2d_conf());
  if (op_conf().conv2d_conf().use_cudnn()) {
#ifndef WITH_CUDNN
    LOG(FATAL);
#endif  // WITH_CUDNN
  }

  EnrollInputBn("in");
  EnrollOutputBn("out");

  EnrollModelBn("filter");
  EnrollModelBn("bias");
  if (!op_conf().conv2d_conf().use_cudnn()) {
    EnrollModelTmpBn("bias_multiplier");
  }

  if (op_conf().conv2d_conf().use_cudnn()) {
    EnrollDataTmpBn("cudnn_workspace");
  } else {
    EnrollDataTmpBn("col_buf");
  }
}

const PbMessage& Conv2dOp::GetSpecialConf() const {
  return op_conf().conv2d_conf();
}

void Conv2dOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const Conv2dOpConf& conf = op_conf().conv2d_conf();
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

  // filter
  BlobDesc* filter_blob_desc = GetBlobDesc4BnInOp("filter");
  filter_blob_desc->mut_shape() =
      Shape({c_o, c_i, conf.kernel_h(), conf.kernel_w()});
  filter_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
  filter_blob_desc->set_has_data_id_field(false);

  // bias
  BlobDesc* bias_blob_desc = GetBlobDesc4BnInOp("bias");
  bias_blob_desc->mut_shape() = Shape({c_o});
  bias_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
  bias_blob_desc->set_has_data_id_field(false);

  if (!conf.use_cudnn()) {
    // bias multiplier
    BlobDesc* bias_multiplier_blob_desc = GetBlobDesc4BnInOp("bias_multiplier");
    bias_multiplier_blob_desc->mut_shape() = Shape({output_size});
    bias_multiplier_blob_desc->set_data_type(
        JobDesc::Singleton()->DefaultDataType());
    bias_multiplier_blob_desc->set_has_data_id_field(false);
  }

#ifdef WITH_CUDNN
  if (conf.use_cudnn()) {
    size_t cudnn_workspace_size =
        InferCudnnWorkspaceSize(GetBlobDesc4BnInOp, conf);

    BlobDesc* cudnn_workspace_blob_desc = GetBlobDesc4BnInOp("cudnn_workspace");
    cudnn_workspace_blob_desc->mut_shape() =
        Shape({static_cast<int64_t>(cudnn_workspace_size)});
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

void Conv2dOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
#ifdef WITH_CUDNN
  if (op_conf().conv2d_conf().use_cudnn()) {
    SetCudnnConvAlgoForKernelConf(GetBlobDesc4BnInOp, op_conf().conv2d_conf(),
                                  kernel_conf->mutable_conv2d_conf());
  }
#endif  // WITH_CUDNN
}

REGISTER_OP(OperatorConf::kConv2DConf, Conv2dOp);

}  // namespace oneflow
