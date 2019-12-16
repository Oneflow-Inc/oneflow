#include "oneflow/core/operator/deconv_op.h"
#include "oneflow/core/kernel/deconv_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

template<typename T>
class DeconvGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvGPUKernel);
  DeconvGPUKernel() = default;
  ~DeconvGPUKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().deconv_conf(); }

  void VirtualKernelInit() override {
    const DeconvOpConf& op_conf = this->op_conf().deconv_conf();
    const ConvConf& conv_conf = this->op_conf().deconv_conf().conv_conf();
    const int32_t num_spatial_dims = this->op_conf().deconv_conf().conv_conf().num_spatial_dims();
    ShapeView x_shape(this->kernel_conf().deconv_conf().in());
    ShapeView y_shape(this->kernel_conf().deconv_conf().out());
    ShapeView weight_shape(this->kernel_conf().deconv_conf().weight());

    const std::string& data_format = conv_conf.data_format();
    this->x_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, x_shape, data_format));
    this->y_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, y_shape, data_format));
    this->filter_desc_.reset(new CudnnFilterDesc(GetDataType<T>::value, weight_shape, data_format));
    this->deconv_desc_.reset(new CudnnDeconvDesc(GetDataType<T>::value, x_shape,
                                                 this->op_conf().deconv_conf().conv_conf()));
    if (op_conf.use_bias()) {
      int32_t filters = op_conf.filters();
      if (num_spatial_dims == 2) {
        if (data_format == "channels_first") {
          this->bias_desc_.reset(
              new CudnnTensorDesc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, 1, filters, 1, 1));
        } else if (data_format == "channels_last") {
          if (GetDataType<T>::value == DataType::kDouble) {
            LOG(FATAL) << "CUDNN 1d & 2d support channels last only if data type "
                          "is float";
          }
          this->bias_desc_.reset(
              new CudnnTensorDesc(CUDNN_TENSOR_NHWC, GetDataType<T>::value, 1, filters, 1, 1));
        } else {
          UNIMPLEMENTED();
        }
      } else {
        if (data_format == "channels_last") {
          LOG(FATAL) << "CUDNN Nd API only support channels first";
        }
        std::vector<int32_t> bias_dim(num_spatial_dims + 2, 1);
        std::vector<int32_t> stride_of_bias_tensor(num_spatial_dims + 2, 1);
        bias_dim[1] = filters;
        stride_of_bias_tensor[0] = filters;
        this->bias_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, num_spatial_dims + 2,
                                                   bias_dim.data(), stride_of_bias_tensor.data()));
      }
    }
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    CudnnConvArgs args(this->op_conf().deconv_conf().conv_conf(), ctx.device_ctx->cudnn_handle(),
                       BnInOp2Blob("y"), BnInOp2Blob("x"), BnInOp2Blob("filter"),
                       BnInOp2Blob("cudnn_buf"),
                       this->job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                       this->job_desc().job_conf().cudnn_conv_heuristic_search_algo(),
                       this->job_desc().cudnn_conv_enable_true_half());
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
    CHECK_LE(work_space_size, BnInOp2Blob("cudnn_buf")->ByteSizeOfBlobBody());
    CudaCheck(cudnnConvolutionBackwardData(args.handle, CudnnSPOnePtr<T>(), args.wdesc.Get(),
                                           args.w_dptr, args.ydesc.Get(), args.y_dptr,
                                           args.cdesc.Get(), algo, args.work_space, work_space_size,
                                           CudnnSPZeroPtr<T>(), args.xdesc.Get(), args.x_dptr));
  }

  mutable std::unique_ptr<CudnnTensorDesc> x_desc_;
  mutable std::unique_ptr<CudnnTensorDesc> y_desc_;
  mutable std::unique_ptr<CudnnFilterDesc> filter_desc_;
  mutable std::unique_ptr<CudnnDeconvDesc> deconv_desc_;
  mutable std::unique_ptr<CudnnTensorDesc> bias_desc_;
};

#define REGISTER_DECONV_GPU_KERNEL(dtype)                                                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kDeconvConf, DeviceType::kGPU, dtype, \
                                        DeconvGPUKernel<dtype>)

REGISTER_DECONV_GPU_KERNEL(float);
REGISTER_DECONV_GPU_KERNEL(double);
REGISTER_DECONV_GPU_KERNEL(float16);

}  //  namespace oneflow
