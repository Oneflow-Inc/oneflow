#ifndef ONEFLOW_CORE_COMMON_EIGEN_UTIL_H_
#define ONEFLOW_CORE_COMMON_EIGEN_UTIL_H_

#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

namespace oneflow {

template<typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
using EigenTensor =
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>;

template<typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
using EigenConstTensor =
    Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>;

template<typename T>
using EigenMatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template<typename T>
using EigenArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

template<typename T>
using ConstEigenMatrixMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template<typename T>
using ConstEigenArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EIGEN_UTIL_H_
