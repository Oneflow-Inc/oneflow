#include "oneflow/core/operator/convolution_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ConvolutionOp::InitFromOpConf() {
  CHECK(op_conf().has_convolution_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");

  EnrollModelBn("weight");
  if (GetBoolFromSpecialConf("has_bias_term")) {
    EnrollModelBn("bias");
#ifndef USE_CUDNN
    EnrollModelTmpBn("bias_multiplier");
#endif
  }

#ifdef USE_CUDNN
  EnrollDataTmpBn("fwd_workspace");
  EnrollDataTmpBn("bwd_weight_workspace");
  EnrollDataTmpBn("bwd_data_workspace");
#else
  EnrollDataTmpBn("col_buf");
#endif  // USE_CUDNN
}

const PbMessage& ConvolutionOp::GetSpecialConf() const {
  return op_conf().convolution_conf();
}

void ConvolutionOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  const ConvolutionOpConf& conf = op_conf().convolution_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ(in_blob_desc->data_type(),
           JobDesc::Singleton()->default_data_type());
  int64_t data_num = in_blob_desc->shape().At(0);
  int64_t c_i = in_blob_desc->shape().At(1);
  int64_t height = in_blob_desc->shape().At(2);
  int64_t width = in_blob_desc->shape().At(3);

  int32_t out_num = GetInt32FromSpecialConf("out_num");
  if (policy == kModelParallel) {
    BalancedSplitter splitter(out_num, parallel_num);
    out_num = splitter.At(parallel_id).size();
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
  out_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  out_blob_desc->set_has_data_id(in_blob_desc->has_data_id());

  // weight
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() = Shape({c_o, c_i * kernel});
  weight_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  weight_blob_desc->set_has_data_id(false);

  if (conf.has_bias_term()) {
    // bias
    BlobDesc* bias_blob_desc = GetBlobDesc4BnInOp("bias");
    bias_blob_desc->mut_shape() = Shape({c_o});
    bias_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
    bias_blob_desc->set_has_data_id(false);

#ifndef USE_CUDNN
    // bias multiplier
    BlobDesc* bias_multiplier_blob_desc = GetBlobDesc4BnInOp("bias_multiplier");
    bias_multiplier_blob_desc->mut_shape() = Shape({output_size});
    bias_multiplier_blob_desc->set_data_type(
        JobDesc::Singleton()->default_data_type());
    bias_multiplier_blob_desc->set_has_data_id(false);
#endif
  }

#ifdef USE_CUDNN
  cudaStream_t cuda_stream;
  cudnnHandle_t cudnn_handle;

  CudaCheck(cudaStreamCreate(&cuda_stream));
  CudaCheck(cudnnCreate(&cudnn_handle));
  CudaCheck(cudnnSetStream(cudnn_handle, cuda_stream));

  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo;
  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_weight_algo;
  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo;

  /* TODO(shiyuan) find algorithm
  cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf[3];
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo_perf[3];
  cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algo_perf[3];
  */

  size_t fwd_workspace_sizes;
  size_t bwd_weight_workspace_sizes;
  size_t bwd_data_workspace_sizes;

  // default
  cudnn_fwd_algo = (cudnnConvolutionFwdAlgo_t)0;  // TODO(shiyuan)
  cudnn_bwd_weight_algo = (cudnnConvolutionBwdFilterAlgo_t)0;
  cudnn_bwd_data_algo = (cudnnConvolutionBwdDataAlgo_t)0;
  fwd_workspace_sizes = 0;
  bwd_weight_workspace_sizes = 0;
  bwd_data_workspace_sizes = 0;

  cudnnTensorDescriptor_t in_desc;
  cudnnTensorDescriptor_t out_desc;
  cudnnFilterDescriptor_t weight_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  CudaCheck(cudnnCreateTensorDescriptor(&in_desc));
  CudaCheck(cudnnCreateTensorDescriptor(&out_desc));
  CudaCheck(cudnnCreateFilterDescriptor(&weight_desc));
  CudaCheck(cudnnCreateConvolutionDescriptor(&conv_desc));

  cudnnDataType_t cudnn_data_type;
  switch (JobDesc::Singleton()->default_data_type()) {
    case kFloat: cudnn_data_type = CUDNN_DATA_FLOAT; break;
    case kDouble: cudnn_data_type = CUDNN_DATA_DOUBLE; break;
    default: UNEXPECTED_RUN();
  }
  CudaCheck(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_data_type, data_num, c_i, height,
                                       width));
  CudaCheck(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_data_type, data_num, c_o, h_len,
                                       w_len));
  CudaCheck(cudnnSetConvolution2dDescriptor(
      conv_desc, conf.pad_h(), conf.pad_w(), conf.stride_h(), conf.stride_w(),
      1, 1, CUDNN_CROSS_CORRELATION));
  CudaCheck(cudnnSetFilter4dDescriptor(weight_desc, cudnn_data_type,
                                       CUDNN_TENSOR_NCHW, data_num, c_o,
                                       conf.kernel_h(), conf.kernel_w()));

  if (conf.has_bias_term()) {
    CudaCheck(cudnnCreateTensorDescriptor(&bias_desc));
    CudaCheck(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                         cudnn_data_type, 1, c_o, 1, 1));
  }

  // TODO(shiyuan) Get algorithm
  size_t workspace_limit_bytes = 8 * 1024 * 1024;

  CudaCheck(cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle, in_desc, weight_desc, conv_desc, out_desc,
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace_limit_bytes,
      &cudnn_fwd_algo));

  CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle, in_desc, out_desc, conv_desc, weight_desc,
      CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes, &cudnn_bwd_weight_algo));

  CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle, weight_desc, out_desc, conv_desc, in_desc,
      CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, workspace_limit_bytes,
      &cudnn_bwd_data_algo));

  mut_op_conf().mutable_convolution_conf()->set_cudnn_fwd_algo(cudnn_fwd_algo);
  mut_op_conf().mutable_convolution_conf()->set_cudnn_bwd_weight_algo(
      cudnn_bwd_weight_algo);
  mut_op_conf().mutable_convolution_conf()->set_cudnn_bwd_data_algo(
      cudnn_bwd_data_algo);

  /* TODO(shiyuan) Find algorithm
  CudaCheck(cudnnFindConvolutionForwardAlgotirhm(
        cudnn_handle, in_desc, weight_desc, conv_desc, out_desc, 3, 3,
        fwd_algo_perf));
  CudaCheck(cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnn_handle, in_desc, out_desc, conv_desc, weight_desc, 3, 3,
        bwd_weight_algo_perf));
  CudaCheck(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn_handle, weight_desc, out_desc, conv_desc, in_desc, 3, 3,
        bwd_data_algo_perf));

  fwd_algo = fwd_algo_perf[0].algo;
  bwd_weight_algo = bwd_weight_algo_perf[0].algo;
  bwd_data_algo = bwd_data_algo_perf[0].algo;
  */

  CudaCheck(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn_handle, in_desc, weight_desc, conv_desc, out_desc, cudnn_fwd_algo,
      &fwd_workspace_sizes));
  CudaCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_handle, in_desc, out_desc, conv_desc, weight_desc,
      cudnn_bwd_weight_algo, &bwd_weight_workspace_sizes));
  CudaCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_handle, weight_desc, out_desc, conv_desc, in_desc,
      cudnn_bwd_data_algo, &bwd_data_workspace_sizes));

  // workspace
  BlobDesc* fwd_workspace_blob_desc = GetBlobDesc4BnInOp("fwd_workspace");
  fwd_workspace_blob_desc->mut_shape() = Shape({fwd_workspace_sizes});
  fwd_workspace_blob_desc->set_data_type(
      JobDesc::Singleton()->default_data_type());
  fwd_workspace_blob_desc->set_has_data_id(false);

  BlobDesc* bwd_weight_workspace_blob_desc =
      GetBlobDesc4BnInOp("bwd_weight_workspace");
  bwd_weight_workspace_blob_desc->mut_shape() =
      Shape({bwd_weight_workspace_sizes});
  bwd_weight_workspace_blob_desc->set_data_type(
      JobDesc::Singleton()->default_data_type());
  bwd_weight_workspace_blob_desc->set_has_data_id(false);

  BlobDesc* bwd_data_workspace_blob_desc =
      GetBlobDesc4BnInOp("bwd_data_workspace");
  bwd_data_workspace_blob_desc->mut_shape() = Shape({bwd_data_workspace_sizes});
  bwd_data_workspace_blob_desc->set_data_type(
      JobDesc::Singleton()->default_data_type());
  bwd_data_workspace_blob_desc->set_has_data_id(false);

  CudaCheck(cudnnDestroyTensorDescriptor(in_desc));
  CudaCheck(cudnnDestroyTensorDescriptor(out_desc));
  CudaCheck(cudnnDestroyConvolutionDescriptor(conv_desc));
  CudaCheck(cudnnDestroyFilterDescriptor(weight_desc));
  if (conf.has_bias_term()) {
    CudaCheck(cudnnDestroyTensorDescriptor(bias_desc));
  }
  CudaCheck(cudaStreamDestroy(cuda_stream));
  CudaCheck(cudnnDestroy(cudnn_handle));
#else
  // col_buf
  BlobDesc* col_buf_blob_desc = GetBlobDesc4BnInOp("col_buf");
  col_buf_blob_desc->mut_shape() = Shape({data_num, output_size, c_i * kernel});
  col_buf_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  col_buf_blob_desc->set_has_data_id(false);
#endif
}

REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);

}  // namespace oneflow
