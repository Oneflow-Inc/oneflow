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
    CudnnConvArgs args(this->op_conf().conv_filter_grad_conf().conv_conf(),
                       ctx.device_ctx->cudnn_handle(), BnInOp2Blob("x"), BnInOp2Blob("dy"),
                       BnInOp2Blob("filter_diff"), BnInOp2Blob("buf"),
                       this->job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                       this->job_desc().job_conf().cudnn_conv_heuristic_search_algo());
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t work_space_size = 0;
    if (this->job_desc().job_conf().has_cudnn_conv_force_bwd_filter_algo()) {
      algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->job_desc().job_conf().cudnn_conv_force_bwd_filter_algo());
      CudaCheck(GetConvWorkspaceSize(args, algo, &work_space_size));
    } else {
      auto algo_perf = FindCudnnConvAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>(args);
      algo = algo_perf->algo;
      work_space_size = algo_perf->memory;
    }
    CHECK_LE(work_space_size, BnInOp2Blob("buf")->ByteSizeOfBlobBody());
    CudaCheck(cudnnConvolutionBackwardFilter(
        args.handle, CudnnSPOnePtr<T>(), args.xdesc.Get(), args.x_dptr, args.ydesc.Get(),
        args.y_dptr, args.cdesc.Get(), algo, args.work_space, work_space_size, CudnnSPZeroPtr<T>(),
        args.wdesc.Get(), args.w_dptr));
  }
};

#define REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(dtype)                                          \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kConvFilterGradConf, DeviceType::kGPU, \
                                        dtype, ConvFilterGradGpuKernel<dtype>);

REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(float);
REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(double);
REGISTER_CONV_FILTER_GRAD_GPU_KERNEL(float16);

}  // namespace oneflow
