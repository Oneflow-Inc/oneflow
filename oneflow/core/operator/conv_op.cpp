#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

namespace {

cudnnConvolutionFwdAlgo_t InferCudnnConvFwdAlgo(
    const cudnnHandle_t& cudnn_handle, CudnnTensorDesc* in_desc,
    CudnnTensorDesc* out_desc, CudnnFilterDesc* filter_desc,
    CudnnConvDesc* conv_desc) {
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
    CudnnConvDesc* conv_desc) {
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
    CudnnConvDesc* conv_desc) {
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo;

  CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle, filter_desc->Get(), out_desc->Get(), conv_desc->Get(),
      in_desc->Get(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
      &cudnn_bwd_data_algo));

  return cudnn_bwd_data_algo;
}

}  // namespace

CudnnConvDesc::~CudnnConvDesc() {
  CudaCheck(cudnnDestroyConvolutionDescriptor(val_));
}

CudnnConvDesc::CudnnConvDesc(const BlobDesc* in_blob_desc, const int kDimSize,
                             const std::vector<int>& dilation_rate,
                             const std::vector<int>& strides,
                             const std::vector<int>& kernel_size,
                             const std::string& data_format,
                             const std::string& padding) {
  CHECK(in_blob_desc->shape().NumAxes() == kDimSize + 2);
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  std::vector<int64_t> in(kDimSize, 0);
  std::vector<int64_t> out(kDimSize, 0);
  std::vector<int> pad_small_side(kDimSize, 0);
  std::vector<int> pad_large_side(kDimSize, 0);

  for (size_t i = 0; i < kDimSize; ++i) {
    in[i] = (data_format == "channel_first")
                ? (in_blob_desc->shape().At(2 + i))
                : (in_blob_desc->shape().At(1 + i));
    GetWindowedOutputSize(in[i], kernel_size[i], dilation_rate[i], strides[i],
                          padding, &out[i], &pad_small_side[i],
                          &pad_large_side[i]);
  }

  CudaCheck(cudnnSetConvolutionNdDescriptor(
      val_, 5, pad_large_side.data(), strides.data(), dilation_rate.data(),
      CUDNN_CROSS_CORRELATION, GetCudnnDataType(in_blob_desc->data_type())));
}

void ConvOp::InitFromOpConf() {
  std::string data_format = GetStringFromCustomizedConf("data_format");
  std::transform(data_format.begin(), data_format.end(), data_format.begin(),
                 ::tolower);
  if (data_format != "channels_last" && data_format != "channels_first") {
    LOG(FATAL) << "Invalid data format in " << op_name();
  }
  SetStringInCustomizedConf("data_format", data_format);

  std::string padding = GetStringFromCustomizedConf("padding");
  std::transform(padding.begin(), padding.end(), padding.begin(), ::tolower);
  if (padding != "same" && padding != "valid") {
    LOG(FATAL) << "Invalid padding method in " << op_name();
  }

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("weight");
  if (GetBoolFromCustomizedConf("use_bias")) { EnrollModelBn("bias"); }
  if (UseCudnn()) { EnrollDataTmpBn("cudnn_workspace"); }
}

void ConvOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, DeviceType device_type,
    std::function<void(OpContext*)> EnrollOpContext) const {
  size_t offset = 0;
  if (GetStringFromCustomizedConf("data_format") == "channel_first") {
    offset = 2;
  } else {
    offset = 1;
  }

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 5);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());

  // out
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t filters = GetInt32FromCustomizedConf("filters");
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(filters, parallel_ctx->parallel_num());
    filters = splitter.At(parallel_ctx->parallel_id()).size();
  }

  std::vector<int64_t> in(KernelDimSize(), 0);
  std::vector<int64_t> out(KernelDimSize(), 0);
  std::vector<int32_t> pad_small_side(KernelDimSize(), 0);
  std::vector<int32_t> pad_large_side(KernelDimSize(), 0);

  for (size_t i = 0; i < KernelDimSize(); ++i) {
    in[i] = in_blob_desc->shape().At(offset + i);
    GetWindowedOutputSize(
        in[i], GetPbRfFromCustomizedConf<int32_t>("kernel_size").Get(i),
        GetPbRfFromCustomizedConf<int32_t>("dilation_rate").Get(i),
        GetPbRfFromCustomizedConf<int32_t>("strides").Get(i),
        GetStringFromCustomizedConf("padding"), &out[i], &pad_small_side[i],
        &pad_large_side[i]);
  }

  std::vector<int64_t> out_shape = {data_num, filters};
  for (size_t i = 0; i < KernelDimSize(); ++i) {
    out_shape.insert(out_shape.begin() + offset + i, out[i]);
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  out_blob_desc->mut_shape() = Shape(out_shape);
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  // weight
  std::vector<int64_t> weight_shape = {filters, in_blob_desc->shape().At(1)};
  for (size_t i = 0; i < KernelDimSize(); ++i) {
    weight_shape.insert(
        weight_shape.begin() + offset + i,
        GetPbRfFromCustomizedConf<int32_t>("kernel_size").Get(i));
  }
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() = Shape(weight_shape);

  // bias
  if (op_conf().conv_3d_conf().use_bias()) {
    GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({filters});
  }

  // cudnn_workspace
  if (UseCudnn()) {
    GetBlobDesc4BnInOp("cudnn_workspace")->mut_shape() =
        Shape({static_cast<int64_t>(
            InferCudnnWorkspaceSize(GetBlobDesc4BnInOp, EnrollOpContext))});
  }
}

void ConvOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const OpContext* op_ctx,
    KernelConf* kernel_conf) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  size_t offset = 0;
  if (GetStringFromCustomizedConf("data_format") == "channel_first") {
    offset = 2;
  } else {
    offset = 1;
  }

  std::vector<int64_t> in(KernelDimSize(), 0);
  std::vector<int64_t> out(KernelDimSize(), 0);
  std::vector<int32_t> pad_small_side(KernelDimSize(), 0);
  std::vector<int32_t> pad_large_side(KernelDimSize(), 0);

  for (size_t i = 0; i < KernelDimSize(); ++i) {
    in[i] = in_blob_desc->shape().At(offset + i);
    GetWindowedOutputSize(
        in[i], GetPbRfFromCustomizedConf<int32_t>("kernel_size").Get(i),
        GetPbRfFromCustomizedConf<int32_t>("dilation_rate").Get(i),
        GetPbRfFromCustomizedConf<int32_t>("strides").Get(i),
        GetStringFromCustomizedConf("padding"), &out[i], &pad_small_side[i],
        &pad_large_side[i]);
    AddInt32ToPbRfInCustomizedKernelConf(kernel_conf, "pad_small_side",
                                         pad_small_side[i]);
    AddInt32ToPbRfInCustomizedKernelConf(kernel_conf, "pad_large_side",
                                         pad_large_side[i]);
  }

  if (UseCudnn()) {
    SetInt32InCustomizedKernelConf(
        kernel_conf, "cudnn_fwd_algo",
        static_cast<const ConvOpCtx*>(op_ctx)->cudnn_fwd_algo());
    SetInt32InCustomizedKernelConf(
        kernel_conf, "cudnn_bwd_filter_algo",
        static_cast<const ConvOpCtx*>(op_ctx)->cudnn_bwd_filter_algo());
    SetInt32InCustomizedKernelConf(
        kernel_conf, "cudnn_bwd_data_algo",
        static_cast<const ConvOpCtx*>(op_ctx)->cudnn_bwd_data_algo());
  }
}

size_t ConvOp::InferCudnnWorkspaceSize(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    std::function<void(OpContext*)> EnrollOpContext) const {
  size_t fwd_workspace_size = 0;
  size_t bwd_filter_workspace_size = 0;
  size_t bwd_data_workspace_size = 0;

  CudaStreamHandle cuda_handle;

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");

  std::vector<int32_t> dilation_rate(KernelDimSize(), 0);
  std::vector<int32_t> strides(KernelDimSize(), 0);
  std::vector<int32_t> kernel_size(KernelDimSize(), 0);
  for (size_t i = 0; i < KernelDimSize(); ++i) {
    dilation_rate[i] =
        GetPbRfFromCustomizedConf<int32_t>("dilation_rate").Get(i);
    strides[i] = GetPbRfFromCustomizedConf<int32_t>("strides").Get(i);
    kernel_size[i] = GetPbRfFromCustomizedConf<int32_t>("kernel_size").Get(i);
  }

  DataType data_type = in_blob_desc->data_type();

  std::vector<int32_t> in_stride(KernelDimSize(), 1);
  std::vector<int32_t> out_stride(KernelDimSize(), 1);
  for (size_t i = KernelDimSize() - 1; i > 0; --i) {
    for (size_t j = KernelDimSize() - 2; j >= 0; --j) {
      in_stride[j] *= in_blob_desc->shape().At(i);
      out_stride[j] *= out_blob_desc->shape().At(i);
    }
  }

  std::vector<int32_t> in_dim(KernelDimSize(), 0);
  std::vector<int32_t> out_dim(KernelDimSize(), 0);
  for (size_t i = 0; i < KernelDimSize(); ++i) {
    in_dim[i] = in_blob_desc->shape().At(i);
    out_dim[i] = out_blob_desc->shape().At(i);
  }

  CudnnTensorDesc in_desc(data_type, in_dim, in_stride);
  CudnnTensorDesc out_desc(data_type, out_dim, out_stride);
  CudnnFilterDesc filter_desc(data_type, weight_blob_desc->shape(),
                              GetStringFromCustomizedConf("data_format"));
  CudnnConvDesc conv_desc(in_blob_desc, KernelDimSize(), dilation_rate, strides,
                          kernel_size,
                          GetStringFromCustomizedConf("data_format"),
                          GetStringFromCustomizedConf("padding"));

  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo =
      InferCudnnConvFwdAlgo(*cuda_handle.cudnn_handle(), &in_desc, &out_desc,
                            &filter_desc, &conv_desc);
  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo =
      InferCudnnConvBwdFilterAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                  &out_desc, &filter_desc, &conv_desc);
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo =
      InferCudnnConvBwdDataAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                &out_desc, &filter_desc, &conv_desc);

  ConvOpCtx* conv_op_ctx = new ConvOpCtx;
  EnrollOpContext(conv_op_ctx);
  conv_op_ctx->set_cudnn_fwd_algo(static_cast<int32_t>(cudnn_fwd_algo));
  conv_op_ctx->set_cudnn_bwd_filter_algo(
      static_cast<int32_t>(cudnn_bwd_filter_algo));
  conv_op_ctx->set_cudnn_bwd_data_algo(
      static_cast<int32_t>(cudnn_bwd_data_algo));

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

}  // namespace oneflow
