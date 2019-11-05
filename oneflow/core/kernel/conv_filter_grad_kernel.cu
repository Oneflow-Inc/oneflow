#include "oneflow/core/kernel/conv_filter_grad_kernel.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

template<typename T>
struct ConvFilterGradKernelUtil<DeviceType::kGPU, T> final {
  static void Compute(DeviceCtx *ctx, const ConvFilterGradKernelConf &kernel_conf,
                      const ConvConf &conf, const Blob *x, const Blob *dy, Blob *filter_diff,
                      Blob *buf, bool deterministic, bool heuristic) {
    CudnnConvArgs args(conf, ctx->cudnn_handle(), x, dy, filter_diff, buf, deterministic,
                       heuristic);
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t work_space_size = 0;
    if (kernel_conf.has_cudnn_bwd_filter_algo()) {
      algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(kernel_conf.cudnn_bwd_filter_algo());
      work_space_size = args.ws_size;
    } else {
      auto algo_perf = FindCudnnConvAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>(args);
      algo = algo_perf->algo;
      work_space_size = algo_perf->memory;
    }
    CudaCheck(cudnnConvolutionBackwardFilter(
        args.handle, CudnnSPOnePtr<T>(), args.xdesc.Get(), args.x_dptr, args.ydesc.Get(),
        args.y_dptr, args.cdesc.Get(), algo, args.work_space, work_space_size, CudnnSPZeroPtr<T>(),
        args.wdesc.Get(), args.w_dptr));
  }
};

#define INSTANTIATE_CONV_FILTER_GRAD_KERNEL_UTIL(type_cpp, type_proto) \
  template struct ConvFilterGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_FILTER_GRAD_KERNEL_UTIL,
                     FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
