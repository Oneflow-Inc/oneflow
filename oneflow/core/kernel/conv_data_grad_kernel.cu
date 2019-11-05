#include "oneflow/core/kernel/conv_data_grad_kernel.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

template<typename T>
struct ConvDataGradKernelUtil<DeviceType::kGPU, T> final {
  static void Compute(DeviceCtx* ctx, const ConvDataGradKernelConf& kernel_conf,
                      const ConvConf& conf, const Blob* dy, const Blob* filter, Blob* dx, Blob* buf,
                      bool deterministic, bool heuristic) {
    CudnnConvArgs args(conf, ctx->cudnn_handle(), dx, dy, filter, buf, deterministic, heuristic);
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t work_space_size = 0;
    if (kernel_conf.has_cudnn_bwd_data_algo()) {
      algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(kernel_conf.cudnn_bwd_data_algo());
      work_space_size = args.ws_size;
    } else {
      auto algo_perf = FindCudnnConvAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>(args);
      algo = algo_perf->algo;
      work_space_size = algo_perf->memory;
    }
    CudaCheck(cudnnConvolutionBackwardData(
      args.handle, CudnnSPOnePtr<T>(), args.wdesc.Get(), args.w_dptr, args.ydesc.Get(),
      args.y_dptr, args.cdesc.Get(), algo, args.work_space, work_space_size,
      CudnnSPZeroPtr<T>(), args.xdesc.Get(), args.x_dptr));
  }
};

#define INSTANTIATE_CONV_DATA_GRAD_KERNEL_UTIL(type_cpp, type_proto) \
  template struct ConvDataGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_DATA_GRAD_KERNEL_UTIL,
                     FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
