#ifndef ONEFLOW_REGISTER_BLOB_H_
#define ONEFLOW_REGISTER_BLOB_H_

#include <functional>
#include "common/shape.h"

namespace oneflow {

template<typename Dtype>
class Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(Dtype dptr, Shape shape_, std::function<void(Dtype)> deleter)
    : dptr_(dptr), shape_(shape), deleter_(deleter) {}
  ~Blob() {deleter_(dptr_)}

  Dtype* mut_dptr() { return dptr_; }
  const Shape& shape() { return shape_; }

  // The Dtype must be double or float
  static_assert(
      std::is_same<Dtype, double>::value || std::is_same<Dtype, float>::value,
      "The Dtype is not double or float!");

 private:
  Dtype* dptr_ ;
  Shape shape_;
  std::function<void(Dtype)> deleter_;
};

}  // namespace oneflow

#endif  // ONEFLOW_REGISTER_BLOB_H_
