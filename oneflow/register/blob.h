#ifndef ONEFLOW_REGISTER_BLOB_H_
#define ONEFLOW_REGISTER_BLOB_H_

#include <functional>
#include "common/shape.h"

namespace oneflow {

template<typename Dtype>
class Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(Dtype* data, Shape shape_, std::function<void(Dtype*)> deleter)
    : data_(data), shape_(shape), deleter_(deleter) {}
  ~Blob() {deleter_(data_)}

  Dtype* mut_data() { return data_; }
  Shape& mut_shape() { return shape_; }
  const Shape& shape() { return shape_; }

 private:
  Dtype* data_;
  Shape shape_;
  std::function<void(Dtype*)> deleter_;
};

}  // namespace oneflow

#endif  // ONEFLOW_REGISTER_BLOB_H_
