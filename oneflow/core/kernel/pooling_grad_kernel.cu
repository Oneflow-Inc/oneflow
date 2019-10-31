#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pooling_grad_kernel.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

namespace {

void InferPoolingParams(const int32_t num_spatial_dims, const PoolingConf& pooling_conf,
                        const DimVector& x_shape_vec, std::vector<int32_t>* pool_size,
                        std::vector<int32_t>* padding, std::vector<int32_t>* strides) {
  std::vector<int32_t> pool_size_3d = Get3DVecInOpConf(pooling_conf.pool_size(), num_spatial_dims);
  std::vector<int32_t> strides_3d = Get3DVecInOpConf(pooling_conf.strides(), num_spatial_dims);
  DimVector y_shape_vec;
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
struct PoolingGradKernelUtil<DeviceType::kGPU, T> final {
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
    const DenseShapeView& x_shape = x_blob->shape();
    const DimVector x_shape_vec = {GetInDim(x_shape, data_format, 0, num_spatial_dims),
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

#define INSTANTIATE_POOLING_GRAD_KERNEL_UTIL(type_cpp, type_proto) \
  template struct PoolingGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_GRAD_KERNEL_UTIL,
                     FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
