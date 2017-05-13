#ifndef ONEFLOW_REGISTER_BLOB_H_
#define ONEFLOW_REGISTER_BLOB_H_

#include <functional>
#include "common/shape.h"

namespace oneflow {

template<typename Dtype>
class Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob() : data_(data), shape_(shape), delete_func_(delete_func) {}
  ~Blob() {delete_func_(data_)}

  Dtype* mut_data() { return data_; }
  Shape& mut_shape() { return shape_; }
  const Shape& shape() { return shape_; }

 private:
  Dtype* data_;
  Shape shape_;
  std::function<void(Dtype*)> delete_func_;
};

}  // namespace oneflow

#endif  // ONEFLOW_REGISTER_BLOB_H_
