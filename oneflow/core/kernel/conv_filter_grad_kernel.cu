#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

template<typename T>
class ConvFilterGradGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradGpuKernel);
  ConvFilterGradGpuKernel() = default;
  ~ConvFilterGradGpuKernel() = default;

 private:
  const PbMessage &GetCustomizedOpConf() const override {
    return this->op_conf().conv_filter_grad_conf();
  }

  void ForwardDataContent(const KernelCtx &ctx,
                          std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    CudnnConvArgs args(this->job_desc().job_conf(),
                       this->op_conf().conv_filter_grad_conf().conv_conf(),
                       ctx.device_ctx->cudnn_handle(), BnInOp2Blob("x"), BnInOp2Blob("dy"),
                       BnInOp2Blob("filter_diff"), BnInOp2Blob("buf"));
    auto algo_perf = FindCudnnConvAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>(&args);
    CHECK_EQ(algo_perf.status, CUDNN_STATUS_SUCCESS);
    CHECK_LE(algo_perf.memory, BnInOp2Blob("buf")->ByteSizeOfBlobBody());
    CudaCheck(cudnnConvolutionBackwardFilter(
        args.handle, CudnnSPOnePtr<T>(), args.xdesc.Get(), args.x_dptr, args.ydesc.Get(),
        args.y_dptr, args.cdesc.Get(), algo_perf.algo, args.ws_dptr, args.params.max_ws_size,
        CudnnSPZeroPtr<T>(), args.wdesc.Get(), args.w_dptr));
  }
};

#define REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(dtype)                                          \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kConvFilterGradConf, DeviceType::kGPU, \
                                        dtype, ConvFilterGradGpuKernel<dtype>);

REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(float);
REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(double);
REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(float16);

}  // namespace oneflow
