#include "oneflow/core/kernel/average_pooling_kernel.h"

namespace oneflow {

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::ForwardOnCPU(
    const Pooling3DCtx& pooling_ctx, const Blob* in_blob,
    Blob* out_blob) const {
  const std::string& data_format = pooling_ctx.kernel_conf().data_format();
  if (data_format == "channels_first") {
    this->ForwardOnCPUWithOrderNCDHW(pooling_ctx, in_blob, out_blob);
  } else if (data_format == "channels_last") {
    this->ForwardOnCPUWithOrderNDHWC(pooling_ctx, in_blob, out_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
T AveragePoolingKernel<DeviceType::kCPU, T>::ForwardInitialize() const {
  return static_cast<T>(0);
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::ForwardProcess(const T& lhs,
                                                               T& rhs) const {
  rhs += lhs;
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::ForwardProcess(
    const int64_t in_col, const int64_t out_col,
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& in_mat,
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat)
    const {
  out_mat.col(out_col) += in_mat.col(in_col);
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::ForwardFinalize(
    const int64_t size, T& out) const {
  out /= size;
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::ForwardFinalize(
    const int64_t size, const int64_t col,
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat)
    const {
  out_mat.col(col) /= size;
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::BackwardOnCPU(
    const Pooling3DCtx& pooling_ctx, const Blob* out_diff_blob,
    const Blob* out_blob, const Blob* in_blob, Blob* in_diff_blob) const {
  const std::string& data_format = pooling_ctx.kernel_conf().data_format();
  if (data_format == "channels_first") {
    this->BackwardOnCPUWithOrderNCDHW(pooling_ctx, out_diff_blob, out_blob,
                                      in_blob, in_diff_blob);
  } else if (data_format == "channels_last") {
    this->BackwardOnCPUWithOrderNDHWC(pooling_ctx, out_diff_blob, out_blob,
                                      in_blob, in_diff_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::BackwardProcessGrad(
    const T& in, const T& out, const T& out_diff, const float scale,
    T& in_diff) const {
  in_diff += (scale * out_diff);
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::BackwardProcessGrad(
    const int64_t out_col, const int64_t in_col, const float scale,
    Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
        out_arr,
    Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
        in_arr,
    Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
        out_diff_arr,
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
        in_diff_arr) const {
  in_diff_arr.col(in_col) += scale * out_diff_arr.col(out_col);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAveragePooling1DConf,
                           AveragePoolingKernel, ARITHMETIC_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAveragePooling2DConf,
                           AveragePoolingKernel, ARITHMETIC_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAveragePooling3DConf,
                           AveragePoolingKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
