#include "oneflow/core/operator/conv_3d_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

namespace {

cudnnConvolutionFwdAlgo_t InferCudnnConvFwdAlgo(
    const cudnnHandle_t& cudnn_handle, CudnnTensorNdDesc* in_desc,
    CudnnTensorNdDesc* out_desc, CudnnFilterNdDesc* filter_desc,
    CudnnConvNdDesc* conv_desc) {
  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo;

  CudaCheck(cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle, in_desc->Get(), filter_desc->Get(), conv_desc->Get(),
      out_desc->Get(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,
      &cudnn_fwd_algo));

  return cudnn_fwd_algo;
}

cudnnConvolutionBwdFilterAlgo_t InferCudnnConvBwdFilterAlgo(
    const cudnnHandle_t& cudnn_handle, CudnnTensorNdDesc* in_desc,
    CudnnTensorNdDesc* out_desc, CudnnFilterNdDesc* filter_desc,
    CudnnConvNdDesc* conv_desc) {
  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo;

  CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle, in_desc->Get(), out_desc->Get(), conv_desc->Get(),
      filter_desc->Get(), CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
      &cudnn_bwd_filter_algo));

  return cudnn_bwd_filter_algo;
}

cudnnConvolutionBwdDataAlgo_t InferCudnnConvBwdDataAlgo(
    const cudnnHandle_t& cudnn_handle, CudnnTensorNdDesc* in_desc,
    CudnnTensorNdDesc* out_desc, CudnnFilterNdDesc* filter_desc,
    CudnnConvNdDesc* conv_desc) {
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo;

  CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle, filter_desc->Get(), out_desc->Get(), conv_desc->Get(),
      in_desc->Get(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
      &cudnn_bwd_data_algo));

  return cudnn_bwd_data_algo;
}

void SetCudnnConvAlgoForKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const Conv3DOpConf& conv_3d_op_conf,
    Conv3DKernelConf* conv_3d_kernel_conf) {
  CudaStreamHandle cuda_handle;

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");

  DataType data_type = in_blob_desc->data_type();

  CudnnTensorNdDesc in_desc(data_type, in_blob_desc->shape());
  CudnnTensorNdDesc out_desc(data_type, out_blob_desc->shape());
  CudnnFilterNdDesc filter_desc(data_type, conv_3d_op_conf.data_format(),
                                weight_blob_desc->shape());
  CudnnConvNdDesc conv_desc(GetBlobDesc4BnInOp, conv_3d_op_conf);

  conv_3d_kernel_conf->set_cudnn_fwd_algo(
      InferCudnnConvFwdAlgo(*cuda_handle.cudnn_handle(), &in_desc, &out_desc,
                            &filter_desc, &conv_desc));
  conv_3d_kernel_conf->set_cudnn_bwd_filter_algo(
      InferCudnnConvBwdFilterAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                  &out_desc, &filter_desc, &conv_desc));
  conv_3d_kernel_conf->set_cudnn_bwd_data_algo(
      InferCudnnConvBwdDataAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                &out_desc, &filter_desc, &conv_desc));
}

size_t InferCudnnWorkspaceSize(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const Conv3DOpConf& conv_3d_op_conf) {
  size_t fwd_workspace_size = 0;
  size_t bwd_filter_workspace_size = 0;
  size_t bwd_data_workspace_size = 0;

  CudaStreamHandle cuda_handle;

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");

  DataType data_type = in_blob_desc->data_type();

  CudnnTensorNdDesc in_desc(data_type, in_blob_desc->shape());
  CudnnTensorNdDesc out_desc(data_type, out_blob_desc->shape());
  CudnnFilterNdDesc filter_desc(data_type, conv_3d_op_conf.data_format(),
                                weight_blob_desc->shape());
  CudnnConvNdDesc conv_desc(GetBlobDesc4BnInOp, conv_3d_op_conf);

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

CudnnConvNdDesc::~CudnnConvNdDesc() {
  CudaCheck(cudnnDestroyConvolutionDescriptor(val_));
}

CudnnConvNdDesc::CudnnConvNdDesc(
    std::function<const BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const Conv3DOpConf& conv_3d_op_conf) {
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t in[3];
  int64_t out[3];
  int32_t dilation_rate[3];
  int32_t strides[3];
  int32_t pad_small_side[3];
  int32_t pad_large_side[3];

  for (size_t i = 0; i < 3; ++i) {
    in[i] = (conv_3d_op_conf.data_format() == "NCDHW")
                ? (in_blob_desc->shape().At(2 + i))
                : (in_blob_desc->shape().At(1 + i));
    dilation_rate[i] = conv_3d_op_conf.dilation_rate(i);
    strides[i] = conv_3d_op_conf.strides(i);
    GetWindowedOutputSize(in[i], conv_3d_op_conf.kernel_size(i),
                          dilation_rate[i], strides[i],
                          conv_3d_op_conf.padding(), &out[i],
                          &pad_small_side[i], &pad_large_side[i]);
  }

  CudaCheck(cudnnSetConvolutionNdDescriptor(
      val_, 5, pad_large_side, strides, dilation_rate, CUDNN_CROSS_CORRELATION,
      GetCudnnDataType(in_blob_desc->data_type())));
}

void Conv3DOp::InitFromOpConf() {
  auto mutable_conv_3d_op_conf = mut_op_conf().mutable_conv_3d_conf();
  std::transform(mutable_conv_3d_op_conf->data_format().begin(),
                 mutable_conv_3d_op_conf->data_format().end(),
                 mutable_conv_3d_op_conf->mutable_data_format(), ::toupper);
  std::transform(mutable_conv_3d_op_conf->padding().begin(),
                 mutable_conv_3d_op_conf->padding().end(),
                 mutable_conv_3d_op_conf->mutable_padding(), ::toupper);
  CHECK(mutable_conv_3d_op_conf->data_format() == "NCDHW"
        || mutable_conv_3d_op_conf->data_format() == "NDHWC");

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("weight");
  if (op_conf().conv_3d_conf().use_bias()) { EnrollModelBn("bias"); }
  if (UseCudnn()) { EnrollDataTmpBn("cudnn_workspace"); }
}

const PbMessage& Conv3DOp::GetSpecialConf() const {
  return op_conf().conv_3d_conf();
}

void Conv3DOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const Conv3DOpConf& conv_3d_op_conf = op_conf().conv_3d_conf();
  std::string data_format = GetStringFromSpecialConf("data_format");

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 5);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());

  // out
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t filters = GetInt32FromSpecialConf("filters");
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(filters, parallel_ctx->parallel_num());
    filters = splitter.At(parallel_ctx->parallel_id()).size();
  }

  int64_t in[3];
  int64_t out[3];
  int32_t pad_small_side[3];
  int32_t pad_large_side[3];

  for (size_t i = 0; i < 3; ++i) {
    in[i] = (conv_3d_op_conf.data_format() == "NCDHW")
                ? (in_blob_desc->shape().At(2 + i))
                : (in_blob_desc->shape().At(1 + i));
    GetWindowedOutputSize(
        in[i], conv_3d_op_conf.kernel_size(i), conv_3d_op_conf.dilation_rate(i),
        conv_3d_op_conf.strides(i), GetStringFromSpecialConf("padding"),
        &out[i], &pad_small_side[i], &pad_large_side[i]);
  }

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  out_blob_desc->mut_shape() =
      (data_format == "NCDHW")
          ? (Shape({data_num, filters, out[0], out[1], out[2]}))
          : (Shape({data_num, out[0], out[1], out[2], filters}));
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  // weight
  GetBlobDesc4BnInOp("weight")->mut_shape() =
      (data_format == "NCDHW")
          ? (Shape({filters, in_blob_desc->shape().At(1),
                    conv_3d_op_conf.kernel_size(0),
                    conv_3d_op_conf.kernel_size(1),
                    conv_3d_op_conf.kernel_size(2)}))
          : (Shape({filters, conv_3d_op_conf.kernel_size(0),
                    conv_3d_op_conf.kernel_size(1),
                    conv_3d_op_conf.kernel_size(2),
                    in_blob_desc->shape().At(4)}));

  // bias
  if (op_conf().conv_3d_conf().use_bias()) {
    GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({filters});
  }

  // cudnn_workspace
  if (UseCudnn()) {
    GetBlobDesc4BnInOp("cudnn_workspace")->mut_shape() =
        Shape({static_cast<int64_t>(
            InferCudnnWorkspaceSize(GetBlobDesc4BnInOp, conv_3d_op_conf))});
  }
}

void Conv3DOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  auto conv_3d_op_conf = op_conf().conv_3d_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");

  int64_t in[3];
  int64_t out[3];
  int32_t pad_small_side[3];
  int32_t pad_large_side[3];

  for (size_t i = 0; i < 3; ++i) {
    in[i] = (conv_3d_op_conf.data_format() == "NCDHW")
                ? (in_blob_desc->shape().At(2 + i))
                : (in_blob_desc->shape().At(1 + i));
    GetWindowedOutputSize(
        in[i], conv_3d_op_conf.kernel_size(i), conv_3d_op_conf.dilation_rate(i),
        conv_3d_op_conf.strides(i), GetStringFromSpecialConf("padding"),
        &out[i], &pad_small_side[i], &pad_large_side[i]);
    kernel_conf->mutable_conv_3d_conf()->add_pad();
    kernel_conf->mutable_conv_3d_conf()->mutable_pad(i)->set_small_side(
        pad_small_side[i]);
    kernel_conf->mutable_conv_3d_conf()->mutable_pad(i)->set_large_side(
        pad_large_side[i]);
  }

  if (UseCudnn()) {
    SetCudnnConvAlgoForKernelConf(GetBlobDesc4BnInOp, op_conf().conv_3d_conf(),
                                  kernel_conf->mutable_conv_3d_conf());
  }
}

REGISTER_OP(OperatorConf::kConv3DConf, Conv3DOp);

}  // namespace oneflow
