#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/tensor_desc.h"

namespace oneflow {

class Shape;

namespace user_op {

class Tensor final {
 public:
  Tensor(const TensorDesc&, char*);
  Tensor(const Shape&, DataType, char*);
  ~Tensor() = default;

  Tensor(const Tensor&);

  const Shape& shape() const { return desc_.shape(); }
  DataType data_type() const { return desc_.data_type(); }

  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>();
    return static_cast<const T*>(dptr_);
  }

  template<typename T = void>
  T* mut_dptr() {
    CheckDataType<T>();
    return static_cast<T*>(dptr_);
  }

 private:
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL,
           (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
            && desc_.data_type() != DataType::kChar && desc_.data_type() != GetDataType<T>::value))
        << desc_.data_type() << " " << GetDataType<T>::value;
  }

  TensorDesc desc_;
  void* dptr_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
