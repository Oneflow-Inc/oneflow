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
    CudnnConvArgs args(this->op_conf().conv_data_grad_conf().conv_conf(),
                       ctx.device_ctx->cudnn_handle(), BnInOp2Blob("dx"), BnInOp2Blob("dy"),
                       BnInOp2Blob("filter"), BnInOp2Blob("buf"),
                       this->job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                       this->job_desc().job_conf().cudnn_conv_heuristic_search_algo());
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t work_space_size = 0;
    if (this->job_desc().job_conf().has_cudnn_conv_force_bwd_data_algo()) {
      algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(
          this->job_desc().job_conf().cudnn_conv_force_bwd_data_algo());
      CudaCheck(GetConvWorkspaceSize(args, algo, &work_space_size));
    } else {
      auto algo_perf = FindCudnnConvAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>(args);
      algo = algo_perf->algo;
      work_space_size = algo_perf->memory;
    }
    CHECK_LE(work_space_size, BnInOp2Blob("buf")->ByteSizeOfBlobBody());
    CudaCheck(cudnnConvolutionBackwardData(args.handle, CudnnSPOnePtr<T>(), args.wdesc.Get(),
                                           args.w_dptr, args.ydesc.Get(), args.y_dptr,
                                           args.cdesc.Get(), algo, args.work_space, work_space_size,
                                           CudnnSPZeroPtr<T>(), args.xdesc.Get(), args.x_dptr));
  }
};

#define REGISTER_CONV_DATA_GRAD_GPU_KERNEL(dtype)                                                 \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kConvDataGradConf, DeviceType::kGPU, dtype, \
                                        ConvDataGradGpuKernel<dtype>);

REGISTER_CONV_DATA_GRAD_GPU_KERNEL(float);
REGISTER_CONV_DATA_GRAD_GPU_KERNEL(double);
REGISTER_CONV_DATA_GRAD_GPU_KERNEL(float16);

}  // namespace oneflow
