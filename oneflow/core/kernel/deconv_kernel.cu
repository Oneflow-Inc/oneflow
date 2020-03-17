#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class DeconvKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvKernel);
  DeconvKernel() = default;
  ~DeconvKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const { return this->op_conf().deconv_conf(); }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* x_blob = BnInOp2Blob("x");
    const Blob* weight_blob = BnInOp2Blob("weight");
    Blob* y_blob = BnInOp2Blob("y");
    Blob* cudnn_buf = BnInOp2Blob("cudnn_buf");
    CudnnConvArgs args(
        this->op_conf().deconv_conf().conv_conf(), y_blob->data_type(), ShapeView(y_blob->shape()),
        weight_blob->data_type(), ShapeView(weight_blob->shape()), x_blob->data_type(),
        ShapeView(x_blob->shape()),
        GetValFromPbMessage<std::string>(this->op_conf().deconv_conf().conv_conf(), "data_format"),
        cudnn_buf->ByteSizeOfBlobBody(),
        this->job_desc().job_conf().cudnn_conv_heuristic_search_algo(),
        this->job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
        this->job_desc().job_conf().cudnn_conv_enable_pseudo_half());
    using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
    using algo_t = cudnnConvolutionBwdDataAlgo_t;
    perf_t algo_perf;
    if (this->job_desc().job_conf().has_cudnn_conv_force_bwd_data_algo()) {
      algo_perf = GetCudnnConvAlgorithmPerference<perf_t>(
          &args,
          static_cast<algo_t>(this->job_desc().job_conf().has_cudnn_conv_force_bwd_data_algo()));
    } else {
      algo_perf = FindCudnnConvAlgorithm<perf_t>(&args);
    }
    CHECK_EQ(algo_perf.status, CUDNN_STATUS_SUCCESS)
        << "op (" << this->op_conf().name()
        << ") find algorithm perference failed. algo: " << algo_perf.algo;
    CHECK_LE(algo_perf.memory, cudnn_buf->ByteSizeOfBlobBody())
        << "op (" << this->op_conf().name() << ") find algorithm " << algo_perf.algo
        << ", need memory " << algo_perf.memory << ", but cudnn_buf_limit_byte is "
        << cudnn_buf->ByteSizeOfBlobBody();
    CudaCheck(cudnnConvolutionBackwardData(
        ctx.device_ctx->cudnn_handle(), CudnnSPOnePtr<T>(), args.wdesc.Get(), weight_blob->dptr(),
        args.ydesc.Get(), x_blob->dptr(), args.cdesc.Get(), algo_perf.algo, cudnn_buf->mut_dptr(),
        args.params.max_ws_size, CudnnSPZeroPtr<T>(), args.xdesc.Get(), y_blob->mut_dptr()));
  }
};

#define REGISTER_DECONV_KERNEL(dev, dtype)                                     \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kDeconvConf, dev, dtype, \
                                        DeconvKernel<dev, dtype>)

REGISTER_DECONV_KERNEL(DeviceType::kGPU, float);
REGISTER_DECONV_KERNEL(DeviceType::kGPU, double);
REGISTER_DECONV_KERNEL(DeviceType::kGPU, float16);

}  //  namespace oneflow
