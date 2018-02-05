#include "oneflow/core/operator/conv2d_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cudnn_util.h"

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
    const Conv2dOpConf& conv2d_op_conf, Conv2dKernelConf* conv2d_kernel_conf) {
  CudaStreamHandle cuda_handle;

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");

  DataType data_type = in_blob_desc->data_type();

  CudnnTensorDesc in_desc(data_type, in_blob_desc->shape());
  CudnnTensorDesc out_desc(data_type, out_blob_desc->shape());
  CudnnFilterDesc filter_desc(data_type, weight_blob_desc->shape());
  CudnnConvolutionDesc conv_desc(GetBlobDesc4BnInOp, conv2d_op_conf);

  conv2d_kernel_conf->set_cudnn_fwd_algo(
      InferCudnnConvFwdAlgo(*cuda_handle.cudnn_handle(), &in_desc, &out_desc,
                            &filter_desc, &conv_desc));
  conv2d_kernel_conf->set_cudnn_bwd_filter_algo(
      InferCudnnConvBwdFilterAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                  &out_desc, &filter_desc, &conv_desc));
  conv2d_kernel_conf->set_cudnn_bwd_data_algo(
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
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");

  DataType data_type = in_blob_desc->data_type();

  CudnnTensorDesc in_desc(data_type, in_blob_desc->shape());
  CudnnTensorDesc out_desc(data_type, out_blob_desc->shape());
  CudnnFilterDesc filter_desc(data_type, weight_blob_desc->shape());
  CudnnConvolutionDesc conv_desc(GetBlobDesc4BnInOp, conv_op_conf);

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

CudnnConvolutionDesc::~CudnnConvolutionDesc() {
  CudaCheck(cudnnDestroyConvolutionDescriptor(val_));
}

CudnnConvolutionDesc::CudnnConvolutionDesc(
    std::function<const BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const Conv2dOpConf& conv2d_op_conf) {
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");

  int pad_h = 0;
  int pad_w = 0;

  std::string padding = conv2d_op_conf.padding();
  std::transform(padding.begin(), padding.end(), padding.begin(), ::toupper);
  if (padding == "SAME") {
    pad_h = (out_blob_desc->shape().At(2) - 1) * conv2d_op_conf.stride_h()
            + conv2d_op_conf.kernel_h() - in_blob_desc->shape().At(2);
    pad_w = (out_blob_desc->shape().At(3) - 1) * conv2d_op_conf.stride_w()
            + conv2d_op_conf.kernel_w() - in_blob_desc->shape().At(2);
    pad_h = std::ceil(pad_h / 2.0);
    pad_w = std::ceil(pad_w / 2.0);
  } else if (padding == "VALID") {
    pad_h = 0;
    pad_w = 0;
  } else {
    UNEXPECTED_RUN();
  }

  CudaCheck(cudnnSetConvolution2dDescriptor(
      val_, pad_h, pad_w, conv2d_op_conf.stride_h(), conv2d_op_conf.stride_w(),
      conv2d_op_conf.dilation_h(), conv2d_op_conf.dilation_w(),
      CUDNN_CROSS_CORRELATION, GetCudnnDataType(in_blob_desc->data_type())));
}
#endif  // WITH_CUDNN

void Conv2dOp::InitFromOpConf() {
  CHECK(op_conf().has_conv2d_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");

  EnrollModelBn("weight");
  EnrollModelBn("bias");

  if (UseCudnn()) {
    EnrollDataTmpBn("cudnn_workspace");
  } else {
    EnrollModelTmpBn("bias_multiplier");
    EnrollDataTmpBn("col_buf");
  }
}

const PbMessage& Conv2dOp::GetSpecialConf() const {
  return op_conf().conv2d_conf();
}

void Conv2dOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const Conv2dOpConf& conv2d_op_conf = op_conf().conv2d_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
  int64_t data_num = in_blob_desc->shape().At(0);
  int64_t c_i = in_blob_desc->shape().At(1);

  int32_t filters = GetInt32FromSpecialConf("filters");
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(filters, parallel_ctx->parallel_num());
    filters = splitter.At(parallel_ctx->parallel_id()).size();
  }
  int64_t c_o = filters;

  int64_t out_height;
  int64_t out_width;

  std::string padding = conv2d_op_conf.padding();
  std::transform(padding.begin(), padding.end(), padding.begin(), ::toupper);
  if (padding == "SAME") {
    out_height = std::ceil(in_blob_desc->shape().At(2)
                           / (conv2d_op_conf.stride_h() * 1.0));
    out_width = std::ceil(in_blob_desc->shape().At(3)
                          / (conv2d_op_conf.stride_w() * 1.0));
  } else if (padding == "VALID") {
    out_height =
        std::ceil((in_blob_desc->shape().At(2) - conv2d_op_conf.kernel_h() + 1)
                  / (conv2d_op_conf.stride_h() * 1.0));
    out_width =
        std::ceil((in_blob_desc->shape().At(3) - conv2d_op_conf.kernel_w() + 1)
                  / (conv2d_op_conf.stride_w() * 1.0));
  } else {
    UNEXPECTED_RUN();
  }

  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  out_blob_desc->mut_shape() = Shape({data_num, c_o, out_height, out_width});
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  // weight
  GetBlobDesc4BnInOp("weight")->mut_shape() =
      Shape({c_o, c_i, conv2d_op_conf.kernel_h(), conv2d_op_conf.kernel_w()});

  // bias
  GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({c_o});

  if (!UseCudnn()) {
    // bias multiplier
    GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() =
        Shape({out_height * out_width});

    // col_buf
    GetBlobDesc4BnInOp("col_buf")->mut_shape() =
        Shape({1, out_height * out_width,
               c_i * conv2d_op_conf.kernel_h() * conv2d_op_conf.kernel_w()});
  }

#ifdef WITH_CUDNN
  if (UseCudnn()) {
    size_t cudnn_workspace_size =
        InferCudnnWorkspaceSize(GetBlobDesc4BnInOp, conv2d_op_conf);

    GetBlobDesc4BnInOp("cudnn_workspace")->mut_shape() =
        Shape({static_cast<int64_t>(cudnn_workspace_size)});
  }
#endif  // WITH_CUDNN
}

void Conv2dOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
#ifdef WITH_CUDNN
  if (UseCudnn()) {
    SetCudnnConvAlgoForKernelConf(GetBlobDesc4BnInOp, op_conf().conv2d_conf(),
                                  kernel_conf->mutable_conv2d_conf());
  }
#endif  // WITH_CUDNN
}

REGISTER_OP(OperatorConf::kConv2DConf, Conv2dOp);

}  // namespace oneflow
