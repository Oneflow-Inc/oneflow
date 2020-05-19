#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

template<typename T>
class ConvDataGradGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvDataGradGpuKernel);
  ConvDataGradGpuKernel() = default;
  ~ConvDataGradGpuKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().conv_data_grad_conf();
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* dy = BnInOp2Blob("dy");
    const Blob* filter = BnInOp2Blob("filter");
    Blob* dx = BnInOp2Blob("dx");
    Blob* buf = BnInOp2Blob("buf");
    const ConvConf& conv_conf = this->op_conf().conv_data_grad_conf().conv_conf();
    CudnnConvArgs args(conv_conf, dx->data_type(), dx->shape(), filter->data_type(),
                       filter->shape(), dy->data_type(), dy->shape(), conv_conf.data_format(),
                       buf->ByteSizeOfBlobBody(),
                       this->job_desc().job_conf().cudnn_conv_heuristic_search_algo(),
                       this->job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                       this->job_desc().job_conf().cudnn_conv_enable_pseudo_half());
    AllocatedCudnnConvResource res(ctx.device_ctx->cudnn_handle(), dx->mut_dptr(),
                                   const_cast<void*>(filter->dptr()), const_cast<void*>(dy->dptr()),
                                   buf->mut_dptr());
    using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
    using algo_t = cudnnConvolutionBwdDataAlgo_t;
    perf_t algo_perf;
    if (this->job_desc().job_conf().has_cudnn_conv_force_bwd_data_algo()) {
      algo_perf = GetCudnnConvAlgorithmPerferenceWithResource<perf_t>(
          &args, &res,
          static_cast<algo_t>(this->job_desc().job_conf().cudnn_conv_force_bwd_data_algo()));
    } else {
      algo_perf = FindCudnnConvAlgorithmWithResource<perf_t>(&args, &res);
    }
    CHECK_EQ(algo_perf.status, CUDNN_STATUS_SUCCESS)
        << "op (" << this->op_conf().name()
        << ") find algorithm perference failed. algo: " << algo_perf.algo;
    CHECK_LE(algo_perf.memory, buf->ByteSizeOfBlobBody())
        << "op (" << this->op_conf().name() << ") find algorithm " << algo_perf.algo
        << ", need memory " << algo_perf.memory << ", but cudnn_buf_limit_byte is "
        << buf->ByteSizeOfBlobBody();
    CudaCheck(cudnnConvolutionBackwardData(
        ctx.device_ctx->cudnn_handle(), CudnnSPOnePtr<T>(), args.wdesc.Get(), filter->dptr(),
        args.ydesc.Get(), dy->dptr(), args.cdesc.Get(), algo_perf.algo, buf->mut_dptr(),
        args.params.max_ws_size, CudnnSPZeroPtr<T>(), args.xdesc.Get(), dx->mut_dptr()));
  }
};

#define REGISTER_CONV_DATA_GRAD_GPU_KERNEL(dtype)                                                 \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kConvDataGradConf, DeviceType::kGPU, dtype, \
                                        ConvDataGradGpuKernel<dtype>);

REGISTER_CONV_DATA_GRAD_GPU_KERNEL(float);
REGISTER_CONV_DATA_GRAD_GPU_KERNEL(double);
REGISTER_CONV_DATA_GRAD_GPU_KERNEL(float16);

}  // namespace oneflow
