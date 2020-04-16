#include "oneflow/customized/utils/pool_util.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

namespace {

std::vector<int32_t> Get3DVec(const std::vector<int32_t>& original_vec, int32_t NDims) {
  std::vector<int32_t> vec;
  FOR_RANGE(uint8_t, dim, 0, 3) {
    int64_t index = static_cast<int64_t>(dim) - (3 - NDims);
    if (index < 0) {
      vec.push_back(1);
    } else {
      vec.push_back(original_vec.at(index));
    }
  }
  return vec;
}

}  // namespace

Params3D::Params3D(const int32_t dim, const Shape& x_shape, const std::string& data_format,
                   const std::string& padding, const std::vector<int32_t>& pool_size,
                   const std::vector<int32_t>& strides) {
  dim_ = dim;
  data_format_ = data_format;
  x_3d_ = {GetInDim(x_shape, data_format, 0, dim), GetInDim(x_shape, data_format, 1, dim),
           GetInDim(x_shape, data_format, 2, dim)};
  pool_size_3d_ = Get3DVec(pool_size, dim);
  strides_3d_ = Get3DVec(strides, dim);
  Get3DOutputSize(x_3d_, pool_size_3d_, strides_3d_, padding, &y_3d_, &padding_before_3d_,
                  &padding_after_3d_);
  if (data_format == "channels_first") {
    channel_num_ = x_shape.At(1);
  } else if (data_format == "channels_last") {
    channel_num_ = x_shape.At(x_shape.NumAxes() - 1);
  } else {
    UNIMPLEMENTED();
  }
  batch_num_ = x_shape.At(0);
}

Shape Params3D::GetYShape() const {
  DimVector y_dim_vec;
  if (dim_ == 1) {
    y_dim_vec = {y_3d_.at(2)};
  } else if (dim_ == 2) {
    y_dim_vec = {y_3d_.at(1), y_3d_.at(2)};
  } else if (dim_ == 3) {
    y_dim_vec = {y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)};
  } else {
    UNIMPLEMENTED();
  }
  if (data_format_ == "channels_first") {
    y_dim_vec.insert(y_dim_vec.begin(), channel_num_);
  } else if (data_format_ == "channels_last") {
    y_dim_vec.insert(y_dim_vec.end(), channel_num_);
  } else {
    UNIMPLEMENTED();
  }
  y_dim_vec.insert(y_dim_vec.begin(), batch_num_);
  return Shape(y_dim_vec);
}

CudnnPoolDesc::CudnnPoolDesc(cudnnPoolingMode_t pooling_mode, int dims, const int* window,
                             const int* padding, const int* stride) {
  CudaCheck(cudnnCreatePoolingDescriptor(&val_));
  CudaCheck(cudnnSetPoolingNdDescriptor(val_, pooling_mode, CUDNN_NOT_PROPAGATE_NAN, dims, window,
                                        padding, stride));
}

CudnnPoolDesc::~CudnnPoolDesc() { CudaCheck(cudnnDestroyPoolingDescriptor(val_)); }

GPUPoolOpKernelState::GPUPoolOpKernelState(const int32_t dim, const std::string& pooling_type,
                                           const Shape& x_shape, const Shape& y_shape,
                                           const std::string& data_format, const DataType& dtype,
                                           const Params3D& params_3d) {
  cudnnPoolingMode_t pooling_mode_;
  if (pooling_type == "AVG") {
    pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  } else if (pooling_type == "MAX") {
    pooling_mode_ = CUDNN_POOLING_MAX;
  } else {
    UNIMPLEMENTED();
  }

  FixedVector pool_size(dim);
  FixedVector padding(dim);
  FixedVector strides(dim);
  FOR_RANGE(int, i, 0, dim) {
    int32_t index_in_3d = i + 3 - dim;
    pool_size[i] = params_3d.pool_size_3d().at(index_in_3d);
    padding[i] = std::max<int>(params_3d.padding_before_3d().at(index_in_3d),
                               params_3d.padding_after_3d().at(index_in_3d));
    strides[i] = params_3d.strides_3d().at(index_in_3d);
  }

  x_desc_.reset(new CudnnTensorDesc(dtype, x_shape, data_format));
  y_desc_.reset(new CudnnTensorDesc(dtype, y_shape, data_format));
  pooling_desc_.reset(
      new CudnnPoolDesc(pooling_mode_, dim, pool_size.data(), padding.data(), strides.data()));
}

template<typename T>
void PoolKernelUtil<T>::CFirstForward(const Params3D& params_3d, const user_op::Tensor* in_blob,
                                      user_op::Tensor* out_blob,
                                      const ForwardInitialize& initialize,
                                      const CFirstProcess& process,
                                      const CFirstFinalize& finalize) {}
template<typename T>
void PoolKernelUtil<T>::CFirstBackward(const Params3D& params_3d,
                                       const user_op::Tensor* out_diff_blob,
                                       const user_op::Tensor* out_blob,
                                       const user_op::Tensor* in_blob,
                                       user_op::Tensor* in_diff_blob,
                                       const CFirstProcessGrad& process) {}
template<typename T>
void PoolKernelUtil<T>::CLastForward(const Params3D& params_3d, const user_op::Tensor* in_blob,
                                     user_op::Tensor* out_blob,
                                     const ForwardInitialize& forward_initialize,
                                     const CLastProcess& process, const CLastFinalize& finalize) {}
template<typename T>
void PoolKernelUtil<T>::CLastBackward(const Params3D& params_3d,
                                      const user_op::Tensor* out_diff_blob,
                                      const user_op::Tensor* out_blob,
                                      const user_op::Tensor* in_blob, user_op::Tensor* in_diff_blob,
                                      const CLastProcessGrad& process) {}

}  // namespace oneflow
