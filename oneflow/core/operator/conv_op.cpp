#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

#ifdef WITH_CUDA

CudnnConvDesc::~CudnnConvDesc() {
  CudaCheck(cudnnDestroyConvolutionDescriptor(val_));
}

CudnnConvDesc::CudnnConvDesc(const BlobDesc* in_blob_desc,
                             const int kernel_dim_size,
                             const std::vector<int>& dilation_rate,
                             const std::vector<int>& strides,
                             const std::vector<int>& kernel_size,
                             const std::string& data_format,
                             const std::string& padding) {
  CHECK_EQ(in_blob_desc->shape().NumAxes(), kernel_dim_size + 2);
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  std::vector<int64_t> in(kernel_dim_size, 0);
  std::vector<int64_t> out(kernel_dim_size, 0);
  std::vector<int> pad_small_side(kernel_dim_size, 0);
  std::vector<int> pad_large_side(kernel_dim_size, 0);

  for (size_t i = 0; i < kernel_dim_size; ++i) {
    in[i] = (data_format == "channels_first")
                ? (in_blob_desc->shape().At(2 + i))
                : (in_blob_desc->shape().At(1 + i));
    GetWindowedOutputSize(in[i], kernel_size[i], dilation_rate[i], strides[i],
                          padding, &out[i], &pad_small_side[i],
                          &pad_large_side[i]);
  }

  CudaCheck(cudnnSetConvolutionNdDescriptor(
      val_, in_blob_desc->shape().NumAxes(), pad_large_side.data(),
      strides.data(), dilation_rate.data(), CUDNN_CROSS_CORRELATION,
      GetCudnnDataType(in_blob_desc->data_type())));
}
#endif  // WITH_CUDA

struct ConvOpCtx : public OpContext {
  int32_t cudnn_fwd_algo;
  int32_t cudnn_bwd_filter_algo;
  int32_t cudnn_bwd_data_algo;
};

void ConvOp::InitFromOpConf() {
  SetBoolInCustomizedConf("use_cudnn_on_gpu", true);
  CHECK(UseCudnn());

  StrFieldTolower("data_format");
  StrFieldTolower("padding");

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
  if (GetStringFromCustomizedConf("data_format") == "channels_first") {
    offset = 2;
  } else {
    offset = 1;
  }

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  CHECK_EQ(in_blob_desc->shape().NumAxes(), KernelDimSize() + 2);
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
  std::vector<int64_t> weight_shape(in_blob_desc->shape().dim_vec());
  weight_shape[0] = filters;
  for (size_t i = 0; i < KernelDimSize(); ++i) {
    weight_shape[offset + i] =
        GetPbRfFromCustomizedConf<int32_t>("kernel_size").Get(i);
  }
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() = Shape(weight_shape);

  // bias
  if (op_conf().conv_3d_conf().use_bias()) {
    GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({filters});
  }

#ifdef WITH_CUDA
  // cudnn_workspace
  if (UseCudnn()) {
    GetBlobDesc4BnInOp("cudnn_workspace")->mut_shape() =
        Shape({static_cast<int64_t>(
            InferCudnnWorkspaceSize(GetBlobDesc4BnInOp, EnrollOpContext))});
  }
#endif  // WITH_CUDA
}

void ConvOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const OpContext* op_ctx,
    KernelConf* kernel_conf) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");

  in_blob_desc->shape().ToProto(
      kernel_conf->mutable_conv_3d_conf()->mutable_in());
  out_blob_desc->shape().ToProto(
      kernel_conf->mutable_conv_3d_conf()->mutable_out());

  size_t offset = 0;
  if (GetStringFromCustomizedConf("data_format") == "channels_first") {
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

#ifdef WITH_CUDA
  if (UseCudnn()) {
    SetInt32InCustomizedKernelConf(
        kernel_conf, "cudnn_fwd_algo",
        static_cast<const ConvOpCtx*>(op_ctx)->cudnn_fwd_algo);
    SetInt32InCustomizedKernelConf(
        kernel_conf, "cudnn_bwd_filter_algo",
        static_cast<const ConvOpCtx*>(op_ctx)->cudnn_bwd_filter_algo);
    SetInt32InCustomizedKernelConf(
        kernel_conf, "cudnn_bwd_data_algo",
        static_cast<const ConvOpCtx*>(op_ctx)->cudnn_bwd_data_algo);
  }
#endif  // WITH_CUDA
}

#ifdef WITH_CUDA
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

  std::vector<int32_t> stride_of_in_tensor(KernelDimSize(), 1);
  std::vector<int32_t> stride_of_out_tensor(KernelDimSize(), 1);
  for (size_t i = KernelDimSize() - 1; i > 0; --i) {
    for (size_t j = KernelDimSize() - 2; j >= 0; --j) {
      stride_of_in_tensor[j] *= in_blob_desc->shape().At(i);
      stride_of_out_tensor[j] *= out_blob_desc->shape().At(i);
    }
  }
  std::vector<int32_t> in_dim(KernelDimSize(), 0);
  std::vector<int32_t> out_dim(KernelDimSize(), 0);
  for (size_t i = 0; i < KernelDimSize(); ++i) {
    in_dim[i] = in_blob_desc->shape().At(i);
    out_dim[i] = out_blob_desc->shape().At(i);
  }

  CudnnTensorDesc in_desc(data_type, in_dim, stride_of_in_tensor);
  CudnnTensorDesc out_desc(data_type, out_dim, stride_of_out_tensor);
  CudnnFilterDesc filter_desc(data_type, weight_blob_desc->shape(),
                              GetStringFromCustomizedConf("data_format"));
  CudnnConvDesc conv_desc(in_blob_desc, KernelDimSize(), dilation_rate, strides,
                          kernel_size,
                          GetStringFromCustomizedConf("data_format"),
                          GetStringFromCustomizedConf("padding"));

  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo;
  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo;
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo;
  CudaCheck(cudnnGetConvolutionForwardAlgorithm(
      *cuda_handle.cudnn_handle(), in_desc.Get(), filter_desc.Get(),
      conv_desc.Get(), out_desc.Get(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,
      &cudnn_fwd_algo));
  CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm(
      *cuda_handle.cudnn_handle(), in_desc.Get(), out_desc.Get(),
      conv_desc.Get(), filter_desc.Get(),
      CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &cudnn_bwd_filter_algo));
  CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm(
      *cuda_handle.cudnn_handle(), filter_desc.Get(), out_desc.Get(),
      conv_desc.Get(), in_desc.Get(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
      0, &cudnn_bwd_data_algo));

  ConvOpCtx* conv_op_ctx = new ConvOpCtx;
  EnrollOpContext(conv_op_ctx);
  conv_op_ctx->cudnn_fwd_algo = static_cast<int32_t>(cudnn_fwd_algo);
  conv_op_ctx->cudnn_bwd_filter_algo =
      static_cast<int32_t>(cudnn_bwd_filter_algo);
  conv_op_ctx->cudnn_bwd_data_algo = static_cast<int32_t>(cudnn_bwd_data_algo);

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
#endif  // WITH_CUDA

}  // namespace oneflow
