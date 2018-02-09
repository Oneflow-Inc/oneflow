#ifndef ONEFLOW_CORE_EIGEN_EIGEN_TENSOR_IMPLEMENT_H_
#define ONEFLOW_CORE_EIGEN_EIGEN_TENSOR_IMPLEMENT_H_

#include "oneflow/core/eigen/tensor_type.h"

namespace oneflow {

template<typename TD>
class EigenTensorImpl final : EigenTensorIf {
 public:
  EigenTensorImpl(TD* val) : val_(val){};
  ~EigenTensorImpl() = default;

  template<typename T>
  EigenTensorIf& operator=(const T& rhs) override {
    *val_ = rhs;
    return *this;
  }

 private:
  TD* val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EIGEN_EIGEN_TENSOR_IMPLEMENT_H_
