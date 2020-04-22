#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/ops/nn_util.h"
#include "oneflow/core/device/cudnn_conv_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

struct ConvCudnnOpKernelState final : public user_op::OpKernelState {
  std::unique_ptr<CudnnTensorDesc> bias_desc;
};

template<typename T>
class ConvGpuFloatingKernel : public user_op::OpKernel {
 public:
  ConvGpuFloatingKernel() = default;
  ~ConvGpuFloatingKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const {
    auto data_format = ctx->GetAttr<std::string>("data_format");
    int32_t filters = ctx->GetAttr<int32_t>("filters");

    std::shared_ptr<ConvCudnnOpKernelState> state;

    const user_op::TensorDesc* bias = ctx->TensorDesc4ArgNameAndIndex("bias", 0);
    if (bias != nullptr) {
      if (data_format == "channels_first") {
        state->bias_desc.reset(
            new CudnnTensorDesc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, 1, filters, 1, 1));
      } else {
        CHECK_EQ("channels_last", data_format);
        CHECK_EQ(DataType::kFloat, GetDataType<T>::value)
            << "CUDNN 1d & 2d support channels last only if data type "
            << "is float";
        state->bias_desc.reset(
            new CudnnTensorDesc(CUDNN_TENSOR_NHWC, GetDataType<T>::value, 1, filters, 1, 1));
      }
    }

    return std::move(state);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const JobDesc& job_desc = ctx->job_desc();
    const user_op::UserOpConfWrapper& user_op_conf = ctx->user_op_conf();

    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* fw_cudnn_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    size_t byte_size_of_buf = fw_cudnn_buf->shape().elem_cnt();
    CudnnConvArgs args(user_op_conf, in->data_type(), in->shape(), weight->data_type(),
                       weight->shape(), out->data_type(), out->shape(),
                       ctx->GetAttr<std::string>("data_format"), byte_size_of_buf,
                       job_desc.job_conf().cudnn_conv_heuristic_search_algo(),
                       job_desc.job_conf().cudnn_conv_use_deterministic_algo_only(),
                       job_desc.job_conf().cudnn_conv_enable_pseudo_half());
    AllocatedCudnnConvResource res(ctx->device_ctx()->cudnn_handle(), const_cast<void*>(in->dptr()),
                                   const_cast<void*>(weight->dptr()), out->mut_dptr(),
                                   fw_cudnn_buf->mut_dptr());
    using perf_t = cudnnConvolutionFwdAlgoPerf_t;
    using algo_t = cudnnConvolutionFwdAlgo_t;
    perf_t algo_perf;
    if (job_desc.job_conf().has_cudnn_conv_force_fwd_algo()) {
      algo_perf = GetCudnnConvAlgorithmPerferenceWithResource<perf_t>(
          &args, &res, static_cast<algo_t>(job_desc.job_conf().cudnn_conv_force_fwd_algo()));
    } else {
      algo_perf = FindCudnnConvAlgorithmWithResource<perf_t>(&args, &res);
    }
    CHECK_EQ(algo_perf.status, CUDNN_STATUS_SUCCESS)
        << "op (" << user_op_conf.op_name()
        << ") find algorithm perference failed. algo: " << algo_perf.algo;
    CHECK_LE(algo_perf.memory, byte_size_of_buf)
        << "op (" << user_op_conf.op_name() << ") find algorithm " << algo_perf.algo
        << ", need memory " << algo_perf.memory << ", but cudnn_buf_limit_byte is "
        << byte_size_of_buf;
    CudaCheck(cudnnConvolutionForward(ctx->device_ctx()->cudnn_handle(), CudnnSPOnePtr<T>(),
                                      args.xdesc.Get(), in->dptr(), args.wdesc.Get(),
                                      weight->dptr(), args.cdesc.Get(), algo_perf.algo,
                                      fw_cudnn_buf->mut_dptr(), args.params.max_ws_size,
                                      CudnnSPZeroPtr<T>(), args.ydesc.Get(), out->mut_dptr()));

    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    ConvCudnnOpKernelState* conv_state = dynamic_cast<ConvCudnnOpKernelState*>(state);
    if (bias != nullptr) {
      CudaCheck(cudnnAddTensor(ctx->device_ctx()->cudnn_handle(), CudnnSPOnePtr<T>(),
                               conv_state->bias_desc->Get(), bias->dptr<T>(), CudnnSPOnePtr<T>(),
                               args.ydesc.Get(), out->mut_dptr<T>()));
    }
  }
};

REGISTER_USER_KERNEL("conv2d")
    .SetCreateFn<ConvGpuFloatingKernel<float>>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      return ctx.device_type() == DeviceType::kGPU
             && ctx.TensorDesc4ArgNameAndIndex("in", 0)->data_type() == DataType::kFloat;
    })
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      const JobDesc& job_desc = ctx->job_desc();
      const user_op::UserOpConfWrapper& user_op_conf = ctx->user_op_conf();
      const auto* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      const auto* weight = ctx->TensorDesc4ArgNameAndIndex("weight", 0);
      const auto* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      // TODO(niuchong): op_conf.cudnn_buf_limit_mbyte?
      size_t workspace_size = job_desc.cudnn_buf_limit_mbyte();
      if (!out->is_dynamic()) {
        CudnnConvArgs args(user_op_conf, in->data_type(), ShapeView(in->shape()),
                           weight->data_type(), ShapeView(weight->shape()), out->data_type(),
                           ShapeView(out->shape()), ctx->GetAttr<std::string>("data_format"),
                           workspace_size, job_desc.job_conf().cudnn_conv_heuristic_search_algo(),
                           job_desc.job_conf().cudnn_conv_use_deterministic_algo_only(),
                           job_desc.job_conf().cudnn_conv_enable_pseudo_half());
        using perf_t = cudnnConvolutionFwdAlgoPerf_t;
        using algo_t = cudnnConvolutionFwdAlgo_t;
        perf_t algo_perf;
        if (job_desc.job_conf().has_cudnn_conv_force_fwd_algo()) {
          algo_perf = GetCudnnConvAlgorithmPerference<perf_t>(
              &args, static_cast<algo_t>(job_desc.job_conf().cudnn_conv_force_fwd_algo()));
        } else {
          algo_perf = FindCudnnConvAlgorithm<perf_t>(&args);
        }
        CHECK_EQ(algo_perf.status, CUDNN_STATUS_SUCCESS)
            << "op (" << user_op_conf.op_name()
            << ") find algorithm perference failed. algo: " << algo_perf.algo;
        CHECK_LE(algo_perf.memory, workspace_size)
            << "op (" << user_op_conf.op_name() << ") find algorithm " << algo_perf.algo
            << ", need memory " << algo_perf.memory << ", but cudnn_buf_limit_byte is "
            << workspace_size;
        workspace_size = algo_perf.memory;
      }
      workspace_size = std::max(size_t(1), workspace_size);
      return workspace_size;
    });

}  // namespace oneflow
