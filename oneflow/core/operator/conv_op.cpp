#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cudnn_conv_util.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

template<int32_t NDims>
void ConvOp<NDims>::InitFromOpConf() {
  StrFieldTolower("data_format");
  StrFieldTolower("padding");
  if (GetValFromCustomizedConf<int32_t>("groups") != 1) {
    CHECK(DevIsGpuAndEnableCudnn()) << "only enable_cudnn support groups > 1";
    CHECK_EQ(GetValFromCustomizedConf<std::string>("data_format"), "channels_first")
        << "only channel_first support groups > 1";
  }
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollInputBn("weight");
  EnrollTmpBn("fw_cudnn_buf");
  EnrollTmpBn("fw_col_buf");
  if (GetValFromCustomizedConf<bool>("use_bias")) {
    CHECK(!GetValFromCustomizedConf<std::string>("bias").empty());
    EnrollInputBn("bias");
    EnrollConstBufBn("bias_multiplier");
  }
}

template<int32_t NDims>
Maybe<void> ConvOp<NDims>::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const std::string& data_format = GetValFromCustomizedConf<std::string>("data_format");

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in_blob_desc->shape().NumAxes(), NDims + 2);
  // CHECK_EQ(in_blob_desc->data_type(), job_desc().DefaultDataType());

  // out
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t filters = GetValFromCustomizedConf<int32_t>("filters");
  // only support data parallel
  CHECK_OR_RETURN(parallel_ctx->parallel_num() == 1
                  || sbp_signature->bn_in_op2sbp_parallel().at("weight").has_broadcast_parallel());

  DimVector out;
  GetConvOutAndPad(in_blob_desc->shape(), GetCustomizedConf(), &out, nullptr, nullptr);
  DimVector out_shape = {data_num, filters};
  size_t dhw_offset = DhwOffset(data_format);
  for (size_t i = 0; i < NDims; ++i) {
    out_shape.insert(out_shape.begin() + dhw_offset + i, out[i]);
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_shape);
  return Maybe<void>::Ok();
}

template<int32_t NDims>
Maybe<void> ConvOp<NDims>::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const std::string& data_format = GetValFromCustomizedConf<std::string>("data_format");

  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in_blob_desc->shape().NumAxes(), NDims + 2);
  // CHECK_EQ(in_blob_desc->data_type(), job_desc().DefaultDataType());

  // out
  int64_t data_num = in_blob_desc->shape().At(0);
  int32_t filters = GetValFromCustomizedConf<int32_t>("filters");

  // only support data parallel
  CHECK_OR_RETURN(parallel_ctx->parallel_num() == 1
                  || sbp_signature->bn_in_op2sbp_parallel().at("weight").has_broadcast_parallel());

  DimVector out;
  GetConvOutAndPad(in_blob_desc->shape(), GetCustomizedConf(), &out, nullptr, nullptr);
  DimVector out_shape = {data_num, filters};
  size_t dhw_offset = DhwOffset(data_format);
  for (size_t i = 0; i < NDims; ++i) {
    out_shape.insert(out_shape.begin() + dhw_offset + i, out[i]);
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_shape);

  // weight
  const BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  DimVector weight_shape(in_blob_desc->shape().dim_vec());
  weight_shape[0] = filters;
  const int32_t groups = GetValFromCustomizedConf<int32_t>("groups");
  CHECK_GT_OR_RETURN(groups, 0);
  CHECK_LE_OR_RETURN(groups, filters);
  CHECK_EQ_OR_RETURN(filters % groups, 0);
  if (data_format == "channels_first") {
    CHECK_LE_OR_RETURN(groups, weight_shape[1]);
    CHECK_EQ_OR_RETURN(weight_shape[1] % groups, 0);
    weight_shape[1] = weight_shape[1] / groups;
  } else if (data_format == "channels_last") {
    CHECK_LE_OR_RETURN(groups, weight_shape[NDims + 1]);
    CHECK_EQ_OR_RETURN(weight_shape[NDims + 1] % groups, 0);
    weight_shape[NDims + 1] = weight_shape[NDims + 1] / groups;
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  for (size_t i = 0; i < NDims; ++i) {
    weight_shape[dhw_offset + i] = GetPbRfFromCustomizedConf<int32_t>("kernel_size").Get(i);
  }
  CHECK_EQ_OR_RETURN(weight_blob_desc->shape(), Shape(weight_shape));

  if (GetValFromCustomizedConf<bool>("use_bias")) {
    // bias and bias_multiplier
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("bias")->shape(), Shape({filters}));
    if (DevIsGpuAndEnableCudnn() == false) {
      DimVector bias_mul_shape(NDims + 1, 1);
      for (size_t i = 0; i != NDims; ++i) { bias_mul_shape[i + 1] = out_shape[dhw_offset + i]; }
      GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() = Shape(bias_mul_shape);
    }
  }

  if (DevIsGpuAndEnableCudnn() == false) {
    // col_buf
    int64_t col_buf_elem_cnt = 1;
    for (size_t i = 0; i != NDims + 1; ++i) { col_buf_elem_cnt *= weight_shape[i + 1]; }
    for (size_t i = 0; i != NDims; ++i) { col_buf_elem_cnt *= out_shape[dhw_offset + i]; }
    int64_t col_buf_size = col_buf_elem_cnt * GetSizeOfDataType(in_blob_desc->data_type());
    BlobDesc* fw_col_buf = GetBlobDesc4BnInOp("fw_col_buf");
    fw_col_buf->mut_shape() = Shape({col_buf_size});
    fw_col_buf->set_data_type(DataType::kChar);
  }

#ifdef WITH_CUDA
  if (DevIsGpuAndEnableCudnn()) {
    // cudnn_buf
    size_t workspace_size = cudnn_buf_limit_byte();
    if (!out_blob_desc->is_dynamic()) {
      CudnnConvArgs args(GetCustomizedConf(), in_blob_desc->data_type(),
                         ShapeView(in_blob_desc->shape()), weight_blob_desc->data_type(),
                         ShapeView(weight_blob_desc->shape()), out_blob_desc->data_type(),
                         ShapeView(out_blob_desc->shape()),
                         GetValFromPbMessage<std::string>(GetCustomizedConf(), "data_format"),
                         workspace_size, job_desc().job_conf().cudnn_conv_heuristic_search_algo(),
                         job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                         job_desc().job_conf().cudnn_conv_enable_pseudo_half());
      using perf_t = cudnnConvolutionFwdAlgoPerf_t;
      using algo_t = cudnnConvolutionFwdAlgo_t;
      perf_t algo_perf;
      if (job_desc().job_conf().has_cudnn_conv_force_fwd_algo()) {
        algo_perf = GetCudnnConvAlgorithmPerference<perf_t>(
            &args, static_cast<algo_t>(this->job_desc().job_conf().cudnn_conv_force_fwd_algo()));
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
    BlobDesc* fw_cudnn_buf = GetBlobDesc4BnInOp("fw_cudnn_buf");
    fw_cudnn_buf->mut_shape() = Shape({static_cast<int64_t>(workspace_size)});
    fw_cudnn_buf->set_data_type(DataType::kChar);
  }
#endif  // WITH_CUDA
  return Maybe<void>::Ok();
}

template<int32_t NDims>
void ConvOp<NDims>::GenKernelConfWithoutCudnn(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    ConvKernelConf* conv_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("in")->shape();
  const Shape& weight_shape = GetBlobDesc4BnInOp("weight")->shape();
  std::string data_format = GetValFromCustomizedConf<std::string>("data_format");
  DimVector in = {GetInDim(in_shape, data_format, 0, NDims),
                  GetInDim(in_shape, data_format, 1, NDims),
                  GetInDim(in_shape, data_format, 2, NDims)};
  DimVector out;
  std::vector<int32_t> weight =
      Get3DVecInOpConf(this->GetPbRfFromCustomizedConf<int32_t>("kernel_size"), NDims);
  std::vector<int32_t> strides =
      Get3DVecInOpConf(this->GetPbRfFromCustomizedConf<int32_t>("strides"), NDims);
  std::vector<int32_t> dilation_rate =
      Get3DVecInOpConf(this->GetPbRfFromCustomizedConf<int32_t>("dilation_rate"), NDims);
  std::vector<int32_t> pad_small_side;
  std::vector<int32_t> pad_large_side;
  Get3DOutputSize(in, weight, strides, GetValFromCustomizedConf<std::string>("padding"), &out,
                  &pad_small_side, &pad_large_side, &dilation_rate);
  FOR_RANGE(size_t, i, 0, 3) {
    conv_conf->mutable_strides()->Add(strides.at(i));
    conv_conf->mutable_pad_small_side()->Add(pad_small_side.at(i));
    conv_conf->mutable_pad_large_side()->Add(pad_large_side.at(i));
    conv_conf->mutable_dilation_rate()->Add(dilation_rate.at(i));
  }
  const Shape& out_shape = GetBlobDesc4BnInOp("out")->shape();
  if (data_format == "channels_first") {
    Shape({in_shape.At(0), in_shape.At(1), in.at(0), in.at(1), in.at(2)})
        .ToProto(conv_conf->mutable_in());
    Shape({out_shape.At(0), out_shape.At(1), out.at(0), out.at(1), out.at(2)})
        .ToProto(conv_conf->mutable_out());
    Shape({weight_shape.At(0), weight_shape.At(1), weight.at(0), weight.at(1), weight.at(2)})
        .ToProto(conv_conf->mutable_weight());
  } else if (data_format == "channels_last") {
    Shape({in_shape.At(0), in.at(0), in.at(1), in.at(2), in_shape.At(in_shape.NumAxes() - 1)})
        .ToProto(conv_conf->mutable_in());
    Shape({out_shape.At(0), out.at(0), out.at(1), out.at(2), out_shape.At(out_shape.NumAxes() - 1)})
        .ToProto(conv_conf->mutable_out());
    Shape({weight_shape.At(0), weight.at(0), weight.at(1), weight.at(2),
           weight_shape.At(weight_shape.NumAxes() - 1)})
        .ToProto(conv_conf->mutable_weight());
  } else {
    UNIMPLEMENTED();
  }
}

template<int32_t NDims>
void ConvOp<NDims>::GenKernelConfWithCudnn(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, KernelConf* kernel_conf,
    ConvKernelConf* conv_conf, const OpContext* op_ctx) const {
  GetBlobDesc4BnInOp("in")->shape().ToProto(conv_conf->mutable_in());
  GetBlobDesc4BnInOp("out")->shape().ToProto(conv_conf->mutable_out());
  GetBlobDesc4BnInOp("weight")->shape().ToProto(conv_conf->mutable_weight());
  if (GetValFromCustomizedConf<bool>("use_bias")) {
    GetBlobDesc4BnInOp("bias")->shape().ToProto(conv_conf->mutable_bias());
  }

  std::vector<int32_t> pad_small_side;
  std::vector<int32_t> pad_large_side;
  GetConvOutAndPad(GetBlobDesc4BnInOp("in")->shape(), GetCustomizedConf(), nullptr, &pad_small_side,
                   &pad_large_side);

  for (size_t i = 0; i < NDims; ++i) {
    AddValToPbRfInCustomizedKernelConf(kernel_conf, "pad_small_side", pad_small_side[i]);
    AddValToPbRfInCustomizedKernelConf(kernel_conf, "pad_large_side", pad_large_side[i]);
  }
}

template<int32_t NDims>
void ConvOp<NDims>::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  ConvKernelConf* conv_conf = kernel_conf->mutable_conv_conf();
  conv_conf->set_dim(NDims);
  if (DevIsGpuAndEnableCudnn()) {
    GenKernelConfWithCudnn(GetBlobDesc4BnInOp, kernel_conf, conv_conf, op_ctx);
  } else {
    GenKernelConfWithoutCudnn(GetBlobDesc4BnInOp, conv_conf);
  }
}

template<int32_t NDims>
PbMessage* ConvOp<NDims>::MutableCustomizedKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_conv_conf();
}

template<int32_t NDims>
Maybe<void> ConvOp<NDims>::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
  return Maybe<void>::Ok();
}

template<int32_t NDims>
Maybe<void> ConvOp<NDims>::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  if (GetValFromCustomizedConf<bool>("use_bias")) {
    SbpSignatureBuilder()
        .Split("in", 0)
        .Broadcast({"weight", "bias"})
        .Split("out", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  } else {
    SbpSignatureBuilder().Split("in", 0).Broadcast("weight").Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

template class ConvOp<1>;
template class ConvOp<2>;
template class ConvOp<3>;

}  // namespace oneflow
