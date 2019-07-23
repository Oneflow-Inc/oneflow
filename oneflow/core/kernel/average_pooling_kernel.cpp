#include "oneflow/core/kernel/average_pooling_kernel.h"

namespace oneflow {

template<typename T>
T AveragePoolingKernel<DeviceType::kCPU, T>::ForwardInitialize() const {
  return GetZeroVal<T>();
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::NCDHWProcess(const T& lhs, T& rhs) const {
  rhs += lhs;
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::NDHWCProcess(const int64_t in_col,
                                                             const int64_t out_col,
                                                             ConstEigenMatrixMap<T>& in_mat,
                                                             EigenMatrixMap<T>& out_mat) const {
  out_mat.col(out_col) += in_mat.col(in_col);
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::NCDHWFinalize(const int64_t size, T& out) const {
  out /= size;
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::NDHWCFinalize(const int64_t size, const int64_t col,
                                                              EigenMatrixMap<T>& out_mat) const {
  out_mat.col(col) /= size;
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::NCDHWProcessGrad(const T& in, const T& out,
                                                                 const T& out_diff,
                                                                 const int64_t size,
                                                                 T& in_diff) const {
  in_diff += (out_diff / static_cast<T>(size));
}

template<typename T>
void AveragePoolingKernel<DeviceType::kCPU, T>::NDHWCProcessGrad(
    const int64_t out_col, const int64_t in_col, const int64_t size, ConstEigenArrayMap<T>& out_arr,
    ConstEigenArrayMap<T>& in_arr, ConstEigenArrayMap<T>& out_diff_arr,
    EigenArrayMap<T>& in_diff_arr) const {
  in_diff_arr.col(in_col) += out_diff_arr.col(out_col) / static_cast<T>(size);
}

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kAveragePooling1DConf, AveragePoolingKernel,
                                         FLOATING_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kAveragePooling2DConf, AveragePoolingKernel,
                                         FLOATING_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kAveragePooling3DConf, AveragePoolingKernel,
                                         FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
