#include "oneflow/core/operator/conv_base_op.h"
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

}  // namespace

CudnnConvNdDesc::~CudnnConvNdDesc() {
  CudaCheck(cudnnDestroyConvolutionDescriptor(val_));
}

CudnnConvNdDesc::CudnnConvNdDesc(const BlobDesc* in_blob_desc,
                                 const int kDimSize,
                                 const std::vector<int32_t>& dilation_rate,
                                 const std::vector<int32_t>& strides,
                                 const std::vector<int32_t>& kernel_size,
                                 const std::string& data_format,
                                 const std::string& padding) {
  CHECK(in_blob_desc->shape().NumAxes() == kDimSize + 2);
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  std::vector<int64_t> in(kDimSize, 0);
  std::vector<int64_t> out(kDimSize, 0);
  std::vector<int32_t> pad_small_side(kDimSize, 0);
  std::vector<int32_t> pad_large_side(kDimSize, 0);

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

void ConvBaseOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const std::string& data_format = GetStringFromSpecialConf("data_format");
  size_t offset = 0;
  if (data_format == "channel_first") {
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
  int32_t filters = GetInt32FromSpecialConf("filters");
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(filters, parallel_ctx->parallel_num());
    filters = splitter.At(parallel_ctx->parallel_id()).size();
  }

  std::vector<int64_t> in(kDimSize, 0);
  std::vector<int64_t> out(kDimSize, 0);
  std::vector<int32_t> pad_small_side(kDimSize, 0);
  std::vector<int32_t> pad_large_side(kDimSize, 0);

  for (size_t i = 0; i < kDimSize; ++i) {
    in[i] = in_blob_desc->shape().At(offset + i);
    GetWindowedOutputSize(in[i],
                          GetPbRfFromSpecialConf("kernel_size").Get(i),
                          GetPbRfFromSpecialConf("dilation_rate").Get(i),
                          GetPbRfFromSpecialConf("strides").Get(i),
                          GetStringFromSpecialConf("padding"), &out[i],
                          &pad_small_side[i], &pad_large_side[i]);
  }

  std::vector<int64_t> out_shape = {data_num, filters};
  for (size_t i = 0; i < kDimSize; ++i) {
    out_blob_desc->mut_shape().insert(offset + i, out[i]);
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  out_blob_desc->mut_shape() = Shape(out_shape);
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  // weight
  std::vector<int64_t> weight_shape = {filters, in_blob_desc->shape().At(1)};
  for (size_t i = 0; i < kDimSize; ++i) {
    weight_blob_desc->mut_shape().insert(
        offset + i, GetPbRfFromSpecialConf("kernel_size").Get(i));
  }
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() = Shape(weight_shape);

  // bias
  if (op_conf().conv_3d_conf().use_bias()) {
    GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({filters});
  }

  // cudnn_workspace
  if (UseCudnn()) {
    GetBlobDesc4BnInOp("cudnn_workspace")->mut_shape() = Shape(
        {static_cast<int64_t>(InferCudnnWorkspaceSize(GetBlobDesc4BnInOp))});
  }
}

void ConvBaseOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");

  std::vector<int64_t> in(kDimSize, 0);
  std::vector<int64_t> out(kDimSize, 0);
  std::vector<int32_t> pad_small_side(kDimSize, 0);
  std::vector<int32_t> pad_large_side(kDimSize, 0);

  for (size_t i = 0; i < kDimSize; ++i) {
    in[i] = in_blob_desc->shape().At(offset + i);
    GetWindowedOutputSize(in[i],
                          GetPbRfFromSpecialConf("kernel_size").Get(i),
                          GetPbRfFromSpecialConf("dilation_rate").Get(i),
                          GetPbRfFromSpecialConf("strides").Get(i),
                          GetStringFromSpecialConf("padding"), &out[i],
                          &pad_small_sidn[i], &pad_large_side[i]);
    MutableConvKernelConf(kernel_conf)->add_pad();
    MutableConvKernelConf(kernel_conf)
        ->mutable_pad(i)
        ->set_small_side(pad_small_side[i]);
    MutableConvKernelConf(kernel_conf)
        ->mutable_pad(i)
        ->set_large_side(pad_large_side[i]);
  }

  if (UseCudnn()) {
    SetCudnnConvAlgoForKernelConf(GetBlobDesc4BnInOp, kernel_conf);
  }
}

void ConvBaseOp::SetCudnnConvAlgoForKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    KernelConf* kernel_conf) {
  CudaStreamHandle cuda_handle;

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");

  std::vector<int32_t> dilation_rate(kDimSize, 0);
  std::vector<int32_t> strides(kDimSize, 0);
  std::vector<int32_t> kernel_size(kDimSize, 0);
  for (size_t i = 0; i < kDimSize; ++i) {
    dilation_rate[i] = GetPbRfFromSpecialConf("dilation_rate").Get(i);
    strides[i] = GetPbRfFromSpecialConf("strides").Get(i);
    kernel_size[i] = GetPbRfFromSpecialConf("kernel_size").Get(i);
  }

  DataType data_type = in_blob_desc->data_type();

  CudnnTensorNdDesc in_desc(data_type, in_blob_desc->shape());
  CudnnTensorNdDesc out_desc(data_type, out_blob_desc->shape());
  CudnnFilterNdDesc filter_desc(data_type,
                                GetInt32FromSpecialConf("data_format"),
                                weight_blob_desc->shape());
  CudnnConvNdDesc conv_desc(in_blob_desc, kDimSize, dilation_rate, strides,
                            kernel_size,
                            GetStringFromSpecialConf("data_format"),
                            GetStringFromSpecialConf("padding"));

  MutableConvKernelConf(kernel_conf)
      ->set_cudnn_fwd_algo(InferCudnnConvFwdAlgo(*cuda_handle.cudnn_handle(),
                                                 &in_desc, &out_desc,
                                                 &filter_desc, &conv_desc));
  MutableConvKernelConf(kernel_conf)
      ->set_cudnn_bwd_filter_algo(
          InferCudnnConvBwdFilterAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                      &out_desc, &filter_desc, &conv_desc));
  MutableConvKernelConf(kernel_conf)
      ->set_cudnn_bwd_data_algo(
          InferCudnnConvBwdDataAlgo(*cuda_handle.cudnn_handle(), &in_desc,
                                    &out_desc, &filter_desc, &conv_desc));
}

size_t ConvBaseOp::InferCudnnWorkspaceSize(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) {
  size_t fwd_workspace_size = 0;
  size_t bwd_filter_workspace_size = 0;
  size_t bwd_data_workspace_size = 0;

  CudaStreamHandle cuda_handle;

  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");

  std::vector<int32_t> dilation_rate(kDimSize, 0);
  std::vector<int32_t> strides(kDimSize, 0);
  std::vector<int32_t> kernel_size(kDimSize, 0);
  for (size_t i = 0; i < kDimSize; ++i) {
    dilation_rate[i] = GetPbRfFromSpecialConf("dilation_rate").Get(i);
    strides[i] = GetPbRfFromSpecialConf("strides").Get(i);
    kernel_size[i] = GetPbRfFromSpecialConf("kernel_size").Get(i);
  }

  DataType data_type = in_blob_desc->data_type();

  CudnnTensorNdDesc in_desc(data_type, in_blob_desc->shape());
  CudnnTensorNdDesc out_desc(data_type, out_blob_desc->shape());
  CudnnFilterNdDesc filter_desc(data_type,
                                GetInt32FromSpecialConf("data_format"),
                                weight_blob_desc->shape());
  CudnnConvNdDesc conv_desc(in_blob_desc, kDimSize, dilation_rate, strides,
                            kernel_size,
                            GetStringFromSpecialConf("data_format"),
                            GetStringFromSpecialConf("padding"));

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

}  // namespace oneflow
