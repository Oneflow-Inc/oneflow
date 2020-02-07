#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

class Blob;

namespace user_op {

class Tensor final {
 public:
  Tensor(Blob*);
  ~Tensor() = default;

  Tensor(const Tensor&);

  const ShapeView& shape() const { return shape_; }
  MutShapeView* mut_shape() { return mut_shape_.get(); }
  DataType data_type() const { return data_type_; }

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
    LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                   && data_type_ != DataType::kChar && data_type_ != GetDataType<T>::value))
        << data_type_ << " " << GetDataType<T>::value;
  }

  void* dptr_;
  ShapeView shape_;
  std::unique_ptr<MutShapeView> mut_shape_;
  DataType data_type_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
