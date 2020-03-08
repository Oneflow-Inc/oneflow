#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

namespace {

void GetDewindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                           int32_t stride, int32_t output_padding, const int32_t padding_needed, int64_t* output_size,
                           int32_t* padding_before, int32_t* padding_after) {
  CHECK_GT(stride, 0);
  CHECK_GE(dilation_rate, 1);

  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  if (output_size) { 
    *output_size = (input_size - 1) * stride + effective_filter_size + output_padding - padding_needed;
    CHECK_GE((*output_size), 0); 
  }
  if (padding_before) { *padding_before = padding_needed / 2; } // not used in deconv
  if (padding_after) { *padding_after = padding_needed - padding_needed / 2;}
}

void GetDeconvOutAndPad(const ShapeView& in_blob_shape, const DeconvOpConf& conf, DimVector* out,
                  std::vector<int32_t>* pad_small_side, std::vector<int32_t>* pad_large_side) {
  const ConvConf& conv_conf = conf.conv_conf();
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  if (out) { out->assign(opkernel_dim, 0); }
  if (pad_small_side) { pad_small_side->assign(opkernel_dim, 0); }
  if (pad_large_side) { pad_large_side->assign(opkernel_dim, 0); }
  const auto& data_format = conv_conf.data_format();
  const auto& strides = conv_conf.strides();
  const PbRf<int32_t>& dilation_rate = conv_conf.dilation_rate();
  const PbRf<int32_t>& kernel_size = conv_conf.kernel_size();
  const PbRf<int32_t>& padding_needed = conv_conf.padding_needed();
  const PbRf<int32_t>& output_padding = conf.output_padding();
  FOR_RANGE(int32_t, i, 0, opkernel_dim) {
    GetDewindowedOutputSize(in_blob_shape.At(DhwOffset(data_format) + i), kernel_size.Get(i), dilation_rate.Get(i),
                            strides.Get(i), output_padding.Get(i), padding_needed.Get(i), out ? &(out->at(i)) : nullptr,
                            pad_small_side ? &(pad_small_side->at(i)) : nullptr,
                            pad_large_side ? &(pad_large_side->at(i)) : nullptr);
  }
}

}  // namespace


class DeconvOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvOp);
  DeconvOp() = default;
  virtual ~DeconvOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().deconv_conf(); }

  void InitFromOpConf() override {
    EnrollInputBn("x");
    EnrollOutputBn("y");
    EnrollInputBn("weight");
    EnrollTmpBn("cudnn_buf");
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override {
    const DeconvOpConf& conf = op_conf().deconv_conf();
    const ConvConf& conv_conf = op_conf().deconv_conf().conv_conf();
    CHECK_OR_RETURN(DevIsGpuAndEnableCudnn()) << "CUDNN is required for Deconv";
    const std::string& data_format = conv_conf.data_format();
    const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
    CHECK_EQ_OR_RETURN(x_blob_desc->shape().NumAxes(), NDims() + 2);
    int64_t data_num = x_blob_desc->shape().At(0);
    int32_t filters = conf.filters();
    DimVector out;
    GetDeconvOutAndPad(x_blob_desc->shape(), conf, &out, nullptr, nullptr);
    DimVector y_shape = {data_num, filters};
    size_t dhw_offset = DhwOffset(data_format);
    for (size_t i = 0; i < NDims(); ++i) {
      y_shape.insert(y_shape.begin() + dhw_offset + i, out[i]);
    }
    BlobDesc* y_blob_desc = GetBlobDesc4BnInOp("y");
    *y_blob_desc = *x_blob_desc;
    y_blob_desc->mut_shape() = Shape(y_shape);

    return Maybe<void>::Ok();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const DeconvOpConf& conf = op_conf().deconv_conf();
    const ConvConf& conv_conf = op_conf().deconv_conf().conv_conf();
    CHECK_OR_RETURN(DevIsGpuAndEnableCudnn()) << "CUDNN is required for Deconv";
    const std::string& data_format = conv_conf.data_format();
    const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
    CHECK_EQ_OR_RETURN(x_blob_desc->shape().NumAxes(), NDims() + 2);
    int64_t data_num = x_blob_desc->shape().At(0);
    int32_t filters = conf.filters();
    // y
    DimVector out;
    GetDeconvOutAndPad(x_blob_desc->shape(), conf, &out, nullptr, nullptr);
    DimVector y_shape = {data_num, filters};
    size_t dhw_offset = DhwOffset(data_format);
    for (size_t i = 0; i < NDims(); ++i) {
      y_shape.insert(y_shape.begin() + dhw_offset + i, out[i]);
    }
    BlobDesc* y_blob_desc = GetBlobDesc4BnInOp("y");
    *y_blob_desc = *x_blob_desc;
    y_blob_desc->mut_shape() = Shape(y_shape);
    // weight
    DimVector weight_shape(y_blob_desc->shape().dim_vec());
    if (data_format == "channels_first") {
      weight_shape[0] = x_blob_desc->shape().At(1);
      weight_shape[1] = filters;
    } else if (data_format == "channels_last") {
      weight_shape[0] = x_blob_desc->shape().At(NDims() + 1);
      weight_shape[NDims() + 1] = filters;
    } else {
      UNIMPLEMENTED();
    }
    for (size_t i = 0; i < NDims(); ++i) {
      weight_shape[dhw_offset + i] = conv_conf.kernel_size(i);
    }
    BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("weight")->shape(), Shape(weight_shape));

#ifdef WITH_CUDA
    if (DevIsGpuAndEnableCudnn()) {
        // cudnn_buf
        size_t workspace_size = cudnn_buf_limit_byte();
        CudnnConvArgs args(conv_conf, y_blob_desc->data_type(),
                         ShapeView(y_blob_desc->shape()), weight_blob_desc->data_type(),
                         ShapeView(weight_blob_desc->shape()), x_blob_desc->data_type(),
                         ShapeView(x_blob_desc->shape()),
                         GetValFromPbMessage<std::string>(conv_conf, "data_format"),
                         workspace_size, job_desc().job_conf().cudnn_conv_heuristic_search_algo(),
                         job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                         job_desc().job_conf().cudnn_conv_enable_pseudo_half());
        using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
        using algo_t = cudnnConvolutionBwdDataAlgo_t;
        perf_t algo_perf;
        if (job_desc().job_conf().has_cudnn_conv_force_bwd_data_algo()) {
            algo_perf = GetCudnnConvAlgorithmPerference<perf_t>(
                &args, static_cast<algo_t>(this->job_desc().job_conf().has_cudnn_conv_force_bwd_data_algo()));
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
        workspace_size = std::max(size_t(1), workspace_size);
        BlobDesc* cudnn_buf = GetBlobDesc4BnInOp("cudnn_buf");
        cudnn_buf->mut_shape() = Shape({static_cast<int64_t>(workspace_size)});
        cudnn_buf->set_data_type(DataType::kChar);
    }
#endif  // WITH_CUDA
    return Maybe<void>::Ok();
  }

 private:
  const int32_t NDims() const { return op_conf().deconv_conf().conv_conf().num_spatial_dims(); }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("y") = *BatchAxis4BnInOp("x");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("x", 0).Broadcast("filter").Split("y", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kDeconvConf, DeconvOp);

}  // namespace oneflow
