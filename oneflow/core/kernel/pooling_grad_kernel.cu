#include "oneflow/core/kernel/pooling_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

namespace {
void InferPoolingParams(const int32_t num_spatial_dims, const PoolingConf& pooling_conf,
                        const std::vector<int64_t> x_shape_vec, std::vector<int32_t>* pool_size,
                        std::vector<int32_t>* padding, std::vector<int32_t>* strides) {
  std::vector<int32_t> pool_size_3d = Get3DVecInOpConf(pooling_conf.pool_size(), num_spatial_dims);
  std::vector<int32_t> strides_3d = Get3DVecInOpConf(pooling_conf.strides(), num_spatial_dims);
  std::vector<int64_t> y_shape_vec;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  Get3DOutputSize(x_shape_vec, pool_size_3d, strides_3d, pooling_conf.padding(), &y_shape_vec,
                  &padding_before, &padding_after);
  FOR_RANGE(int, i, 0, num_spatial_dims) {
    int32_t index_in_3d = i + 3 - num_spatial_dims;
    pool_size->at(i) = pool_size_3d.at(index_in_3d);
    padding->at(i) = std::max(padding_before.at(index_in_3d), padding_after.at(index_in_3d));
    strides->at(i) = strides_3d.at(index_in_3d);
  }
}
}  // namespace

template<typename T>
struct PoolingGradKernelUtil final {
  static void Compute(DeviceCtx* ctx, const PoolingConf& pooling_conf, const Blob* dy_blob,
                      const Blob* y_blob, const Blob* x_blob, Blob* dx_blob) {
    std::string data_format = pooling_conf.data_format();
    CudnnTensorDesc x_desc(x_blob->data_type(), x_blob->shape(), data_format);
    CudnnTensorDesc y_desc(y_blob->data_type(), y_blob->shape(), data_format);
    cudnnPoolingMode_t pooling_mode;
    if (pooling_conf.pool_mode() == "avg") {
      pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    } else if (pooling_conf.pool_mode() == "max") {
      pooling_mode = CUDNN_POOLING_MAX;
    }
    const int32_t num_spatial_dims = pooling_conf.num_spatial_dims();
    std::vector<int> pool_size(num_spatial_dims);
    std::vector<int> padding(num_spatial_dims);
    std::vector<int> strides(num_spatial_dims);
    const Shape& x_shape = x_blob->shape();
    const std::vector<int64_t> x_shape_vec = {GetInDim(x_shape, data_format, 0, num_spatial_dims),
                                              GetInDim(x_shape, data_format, 1, num_spatial_dims),
                                              GetInDim(x_shape, data_format, 2, num_spatial_dims)};
    InferPoolingParams(num_spatial_dims, pooling_conf, x_shape_vec, &pool_size, &padding, &strides);
    std::unique_ptr<CudnnPoolingDesc> pooling_desc;
    pooling_desc.reset(new CudnnPoolingDesc(pooling_mode, num_spatial_dims, pool_size.data(),
                                            padding.data(), strides.data()));
    CudaCheck(cudnnPoolingBackward(ctx->cudnn_handle(), pooling_desc->Get(), CudnnSPOnePtr<T>(),
                                   y_desc.Get(), y_blob->dptr(), y_desc.Get(), dy_blob->dptr(),
                                   x_desc.Get(), x_blob->dptr(), CudnnSPZeroPtr<T>(), x_desc.Get(),
                                   dx_blob->mut_dptr()));
  }
};

template<typename T>
class PoolingGradKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingGradKernel);
  PoolingGradKernel() = default;
  ~PoolingGradKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().pooling_grad_conf();
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* dx_blob = BnInOp2Blob("dx");
    if (dx_blob == nullptr) { return; }
    const PoolingConf& pooling_conf = this->op_conf().pooling_grad_conf().pooling_conf();
    if (pooling_conf.pool_mode() == "max") {
      Memset<DeviceType::kGPU>(ctx.device_ctx, dx_blob->mut_dptr(), 0,
                               dx_blob->ByteSizeOfBlobBody());
    }
    PoolingGradKernelUtil<T>::Compute(ctx.device_ctx, pooling_conf, BnInOp2Blob("dy"),
                                      BnInOp2Blob("y"), BnInOp2Blob("x"), dx_blob);
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kPoolingGradConf, DeviceType::kGPU, float,
                                      PoolingGradKernel<float>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kPoolingGradConf, DeviceType::kGPU, double,
                                      PoolingGradKernel<double>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kPoolingGradConf, DeviceType::kGPU, float16,
                                      PoolingGradKernel<float16>);

}  // namespace oneflow
