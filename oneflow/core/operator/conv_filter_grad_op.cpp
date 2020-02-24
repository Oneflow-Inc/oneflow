#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/conv_filter_grad_op.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

const PbMessage& ConvFilterGradOp::GetCustomizedConf() const {
  return op_conf().conv_filter_grad_conf();
}

void ConvFilterGradOp::InitFromOpConf() {
  CHECK(op_conf().has_conv_filter_grad_conf());
  EnrollInputBn("dy", false);
  EnrollInputBn("x", false);
  EnrollOutputBn("filter_diff", false);
  if (DevIsGpuAndEnableCudnn()) {
    EnrollTmpBn("buf");
  } else {
    UNIMPLEMENTED();
  }
}

Maybe<void> ConvFilterGradOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const ConvFilterGradOpConf& conf = this->op_conf().conv_filter_grad_conf();
  const ConvConf& conv_conf = conf.conv_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  BlobDesc* filter_diff = GetBlobDesc4BnInOp("filter_diff");
  const int32_t num_spatial_dims = conf.conv_conf().num_spatial_dims();
  CHECK_GE_OR_RETURN(num_spatial_dims, 1);
  CHECK_LE_OR_RETURN(num_spatial_dims, 3);
  CHECK_EQ_OR_RETURN(dy->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x->data_type(), dy->data_type());
  DimVector filter_diff_dim_vec;
  if (conv_conf.data_format() == "channels_first") {
    filter_diff_dim_vec.push_back(dy->shape().At(1));
    filter_diff_dim_vec.push_back(x->shape().At(1));
    filter_diff_dim_vec.insert(filter_diff_dim_vec.end(), conv_conf.kernel_size().cbegin(),
                               conv_conf.kernel_size().cend());
  } else if (conv_conf.data_format() == "channels_last") {
    filter_diff_dim_vec.push_back(dy->shape().dim_vec().back());
    filter_diff_dim_vec.insert(filter_diff_dim_vec.end(), conv_conf.kernel_size().cbegin(),
                               conv_conf.kernel_size().cend());
    filter_diff_dim_vec.push_back(x->shape().dim_vec().back());
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  filter_diff->mut_shape() = Shape(filter_diff_dim_vec);
  filter_diff->set_data_type(x->data_type());
  return Maybe<void>::Ok();
}

Maybe<void> ConvFilterGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const ConvFilterGradOpConf& conf = this->op_conf().conv_filter_grad_conf();
  const ConvConf& conv_conf = conf.conv_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  BlobDesc* filter_diff = GetBlobDesc4BnInOp("filter_diff");
  const int32_t num_spatial_dims = conf.conv_conf().num_spatial_dims();
  CHECK_GE_OR_RETURN(num_spatial_dims, 1);
  CHECK_LE_OR_RETURN(num_spatial_dims, 3);
  CHECK_EQ_OR_RETURN(dy->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x->data_type(), dy->data_type());
  DimVector filter_diff_dim_vec;
  if (conv_conf.data_format() == "channels_first") {
    filter_diff_dim_vec.push_back(dy->shape().At(1));
    filter_diff_dim_vec.push_back(x->shape().At(1));
    filter_diff_dim_vec.insert(filter_diff_dim_vec.end(), conv_conf.kernel_size().cbegin(),
                               conv_conf.kernel_size().cend());
  } else if (conv_conf.data_format() == "channels_last") {
    filter_diff_dim_vec.push_back(dy->shape().dim_vec().back());
    filter_diff_dim_vec.insert(filter_diff_dim_vec.end(), conv_conf.kernel_size().cbegin(),
                               conv_conf.kernel_size().cend());
    filter_diff_dim_vec.push_back(x->shape().dim_vec().back());
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  filter_diff->mut_shape() = Shape(filter_diff_dim_vec);
  filter_diff->set_data_type(x->data_type());

  if (DevIsGpuAndEnableCudnn()) {
#ifdef WITH_CUDA
    size_t workspace_size = cudnn_buf_limit_byte();
    if (!x->is_dynamic()) {
      CudnnConvArgs args(conv_conf, x->data_type(), ShapeView(x->shape()), filter_diff->data_type(),
                         ShapeView(filter_diff->shape()), dy->data_type(), ShapeView(dy->shape()),
                         conv_conf.data_format(), workspace_size,
                         job_desc().job_conf().cudnn_conv_heuristic_search_algo(),
                         job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                         job_desc().job_conf().cudnn_conv_enable_pseudo_half());
      using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
      using algo_t = cudnnConvolutionBwdFilterAlgo_t;
      perf_t algo_perf;
      if (this->job_desc().job_conf().has_cudnn_conv_force_bwd_filter_algo()) {
        algo_perf = GetCudnnConvAlgorithmPerference<perf_t>(
            &args,
            static_cast<algo_t>(this->job_desc().job_conf().cudnn_conv_force_bwd_filter_algo()));
      } else {
        algo_perf = FindCudnnConvAlgorithm<perf_t>(&args);
      }
      CHECK_EQ_OR_RETURN(algo_perf.status, CUDNN_STATUS_SUCCESS)
          << "op (" << op_conf().name()
          << ") find algorithm perference failed. algo: " << algo_perf.algo;
      CHECK_LE_OR_RETURN(algo_perf.memory, workspace_size)
          << "op (" << op_conf().name() << ") find algorithm " << algo_perf.algo << ", need memory "
          << algo_perf.memory << ", but cudnn_buf_limit_byte is " << workspace_size;
      workspace_size = algo_perf.memory;
    }
    workspace_size = std::max(size_t(1), workspace_size);
    BlobDesc* cudnn_buf = GetBlobDesc4BnInOp("buf");
    cudnn_buf->set_data_type(DataType::kChar);
    cudnn_buf->mut_shape() = Shape({static_cast<int64_t>(workspace_size)});
#else
    UNIMPLEMENTED_THEN_RETURN();
#endif
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}

Maybe<void> ConvFilterGradOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_OR_RETURN(*BatchAxis4BnInOp("dy") == *BatchAxis4BnInOp("x"));
  BatchAxis4BnInOp("filter_diff")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> ConvFilterGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("dy", 0)
      .Split("x", 0)
      .PartialSum("filter_diff")
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kConvFilterGradConf, ConvFilterGradOp);

}  // namespace oneflow
