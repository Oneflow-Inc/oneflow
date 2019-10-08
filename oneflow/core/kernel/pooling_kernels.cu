#include "oneflow/core/kernel/pooling_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

CudnnPoolingDesc::~CudnnPoolingDesc() { CudaCheck(cudnnDestroyPoolingDescriptor(val_)); }

CudnnPoolingDesc::CudnnPoolingDesc(cudnnPoolingMode_t pooling_mode, int dims, const int* window,
                                   const int* padding, const int* stride) {
  CudaCheck(cudnnCreatePoolingDescriptor(&val_));
  CudaCheck(cudnnSetPoolingNdDescriptor(val_, pooling_mode, CUDNN_NOT_PROPAGATE_NAN, dims, window,
                                        padding, stride));
}

class PoolingGpuCtx final {
 public:
  PoolingGpuCtx(const PoolingKernelConf& kernel_conf, cudnnPoolingMode_t pooling_mode,
                DataType type)
      : kernel_conf_(kernel_conf), pooling_mode_(pooling_mode) {
    int32_t dim = kernel_conf_.dim();
    CHECK_GE(dim, 1);
    CHECK_LE(dim, 3);
    std::vector<int> in_dim = GetStdVecFromShapeInKernelConf("in");
    std::vector<int> out_dim = GetStdVecFromShapeInKernelConf("out");

    std::vector<int> pool_size(dim);
    std::vector<int> padding(dim);
    std::vector<int> strides(dim);
    FOR_RANGE(int, i, 0, dim) {
      int32_t index_in_3d = i + 3 - dim;
      pool_size[i] = kernel_conf_.pool_size().Get(index_in_3d);
      padding[i] = std::max(kernel_conf_.padding_before().Get(index_in_3d),
                            kernel_conf_.padding_after().Get(index_in_3d));
      strides[i] = kernel_conf_.strides().Get(index_in_3d);
    }
    pooling_desc_.reset(
        new CudnnPoolingDesc(pooling_mode_, dim, pool_size.data(), padding.data(), strides.data()));

    int32_t ncx_dim = 2 + dim;
    std::vector<int> in_shape(ncx_dim);
    std::vector<int> out_shape(ncx_dim);
    std::vector<int> in_stride(ncx_dim);
    std::vector<int> out_stride(ncx_dim);

    FOR_RANGE(size_t, i, 0, 2) {
      in_shape[i] = in_dim[i];
      out_shape[i] = out_dim[i];
    }
    FOR_RANGE(int, i, 0, dim) {
      int32_t index_in_3d = 2 + i + 3 - dim;
      in_shape[i + 2] = in_dim[index_in_3d];
      out_shape[i + 2] = out_dim[index_in_3d];
    }

    const std::string& data_format = kernel_conf_.data_format();
    if (data_format == "channels_first") {
      in_stride[ncx_dim - 1] = 1;
      out_stride[ncx_dim - 1] = 1;

      for (int i = ncx_dim - 2; i >= 0; --i) {
        in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
      }
    } else if (data_format == "channels_last") {
      in_stride[ncx_dim - 1] = in_shape[1];
      out_stride[ncx_dim - 1] = out_shape[1];
      for (int i = ncx_dim - 2; i >= 2; --i) {
        in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
      }
      in_stride[1] = 1;
      out_stride[1] = 1;
      in_stride[0] = in_shape[2] * in_stride[2];
      out_stride[0] = out_shape[2] * out_stride[2];
    } else {
      UNIMPLEMENTED();
    }
    in_desc_.reset(new CudnnTensorDesc(type, ncx_dim, in_shape.data(), in_stride.data()));
    out_desc_.reset(new CudnnTensorDesc(type, ncx_dim, out_shape.data(), out_stride.data()));
  }
  ~PoolingGpuCtx() = default;

  const PoolingKernelConf& kernel_conf() const { return kernel_conf_; }

  const cudnnTensorDescriptor_t& cudnn_in_tensor_desc() const { return in_desc_->Get(); }
  const cudnnTensorDescriptor_t& cudnn_out_tensor_desc() const { return out_desc_->Get(); }
  const cudnnPoolingDescriptor_t& cudnn_pooling_desc() const { return pooling_desc_->Get(); }

 private:
  std::vector<int> GetStdVecFromShapeInKernelConf(const std::string& field_name) const {
    const PbRf<int64_t>& shape = GetPbRfFromPbMessage<int64_t>(
        GetValFromPbMessage<const PbMessage&>(kernel_conf_, field_name), "dim");
    std::vector<int> ret(shape.begin(), shape.end());
    return ret;
  }

  PoolingKernelConf kernel_conf_;

  cudnnPoolingMode_t pooling_mode_;
  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> out_desc_;
  std::unique_ptr<CudnnPoolingDesc> pooling_desc_;
};

template<typename T>
class PoolingGpuKernelIf : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingGpuKernelIf);
  PoolingGpuKernelIf() = default;
  virtual ~PoolingGpuKernelIf() = default;

 protected:
  virtual cudnnPoolingMode_t GetCudnnPoolingMode() = 0;
  const PoolingGpuCtx& pooling_ctx() const { return *pooling_ctx_; }
  void VirtualKernelInit() override {
    pooling_ctx_.reset(new PoolingGpuCtx(GetPoolingKernelConf(), this->GetCudnnPoolingMode(),
                                         GetDataType<T>::value));
  }
  virtual const PoolingKernelConf& GetPoolingKernelConf() const = 0;
  void ForwardDataContent(const KernelCtx& kernel_ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    PoolingForward(kernel_ctx, this->pooling_ctx(), in_blob, out_blob);
  }
  virtual void PoolingForward(const KernelCtx& kernel_ctx, const PoolingGpuCtx& pooling_ctx,
                              const Blob* in_blob, Blob* out_blob) const = 0;
  virtual void PoolingBackward(const KernelCtx& kernel_ctx, const PoolingGpuCtx& pooling_ctx,
                               const Blob* out_diff_blob, const Blob* out_blob, const Blob* in_blob,
                               Blob* in_diff_blob) const = 0;

  std::unique_ptr<PoolingGpuCtx> pooling_ctx_;
};

template<typename T>
class PoolingGpuKernel : public PoolingGpuKernelIf<T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingGpuKernel);
  PoolingGpuKernel() = default;
  virtual ~PoolingGpuKernel() = default;

 protected:
  void PoolingForward(const KernelCtx& kernel_ctx, const PoolingGpuCtx& pooling_ctx,
                      const Blob* in_blob, Blob* out_blob) const override {
    CudaCheck(cudnnPoolingForward(
        kernel_ctx.device_ctx->cudnn_handle(), pooling_ctx.cudnn_pooling_desc(), CudnnSPOnePtr<T>(),
        pooling_ctx.cudnn_in_tensor_desc(), in_blob->dptr(), CudnnSPZeroPtr<T>(),
        pooling_ctx.cudnn_out_tensor_desc(), out_blob->mut_dptr()));
  }
  void PoolingBackward(const KernelCtx& kernel_ctx, const PoolingGpuCtx& pooling_ctx,
                       const Blob* out_diff_blob, const Blob* out_blob, const Blob* in_blob,
                       Blob* in_diff_blob) const override {
    CudaCheck(cudnnPoolingBackward(
        kernel_ctx.device_ctx->cudnn_handle(), pooling_ctx.cudnn_pooling_desc(), CudnnSPOnePtr<T>(),
        pooling_ctx.cudnn_out_tensor_desc(), out_blob->dptr(), pooling_ctx.cudnn_out_tensor_desc(),
        out_diff_blob->dptr(), pooling_ctx.cudnn_in_tensor_desc(), in_blob->dptr(),
        CudnnSPZeroPtr<T>(), pooling_ctx.cudnn_in_tensor_desc(), in_diff_blob->mut_dptr()));
  }
};

template<typename T>
class MaxPoolingGpuKernel final : public PoolingGpuKernel<T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingGpuKernel);
  MaxPoolingGpuKernel() = default;
  ~MaxPoolingGpuKernel() = default;

 private:
  const PoolingKernelConf& GetPoolingKernelConf() const override {
    return this->kernel_conf().max_pooling_conf().pooling_conf();
  }
  cudnnPoolingMode_t GetCudnnPoolingMode() override { return CUDNN_POOLING_MAX; }
};

#define REGISTER_POOLING_KERNEL(op_type, dtype, kernel) \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kGPU, dtype, kernel<dtype>);

#define REGISTER_MAX_POOLING_KERNEL(dim)                                                      \
  REGISTER_POOLING_KERNEL(OperatorConf::kMaxPooling##dim##Conf, float, MaxPoolingGpuKernel);  \
  REGISTER_POOLING_KERNEL(OperatorConf::kMaxPooling##dim##Conf, double, MaxPoolingGpuKernel); \
  REGISTER_POOLING_KERNEL(OperatorConf::kMaxPooling##dim##Conf, float16, MaxPoolingGpuKernel);

REGISTER_MAX_POOLING_KERNEL(1D);
REGISTER_MAX_POOLING_KERNEL(2D);
REGISTER_MAX_POOLING_KERNEL(3D);

template<typename T>
class AveragePoolingGpuKernel final : public PoolingGpuKernel<T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingGpuKernel);
  AveragePoolingGpuKernel() = default;
  ~AveragePoolingGpuKernel() = default;

 private:
  const PoolingKernelConf& GetPoolingKernelConf() const override {
    return this->kernel_conf().average_pooling_conf().pooling_conf();
  }
  cudnnPoolingMode_t GetCudnnPoolingMode() override {
    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
};

#define REGISTER_AVERAGE_POOLING_KERNEL(dim)                                 \
  REGISTER_POOLING_KERNEL(OperatorConf::kAveragePooling##dim##Conf, float,   \
                          AveragePoolingGpuKernel);                          \
  REGISTER_POOLING_KERNEL(OperatorConf::kAveragePooling##dim##Conf, double,  \
                          AveragePoolingGpuKernel);                          \
  REGISTER_POOLING_KERNEL(OperatorConf::kAveragePooling##dim##Conf, float16, \
                          AveragePoolingGpuKernel);

REGISTER_AVERAGE_POOLING_KERNEL(1D);
REGISTER_AVERAGE_POOLING_KERNEL(2D);
REGISTER_AVERAGE_POOLING_KERNEL(3D);

}  // namespace oneflow
