#ifndef ONEFLOW_REGISTER_BLOB_H_
#define ONEFLOW_REGISTER_BLOB_H_

#include <functional>
#include "common/shape.h"

namespace oneflow {

class Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(void* dptr, const Shape& shape, std::function<void(void*)> deleter)
    : dptr_(dptr), shape_(shape), deleter_(deleter) {}
  ~Blob() { deleter_(dptr_); }

  const void* dptr() const { return dptr_; }
  const Shape& shape() const { return shape_; }
  
  void* mut_dptr() { return dptr_; }

 private:
  void* dptr_ ;
  Shape shape_;
  std::function<void(void*)> deleter_;
};

}  // namespace oneflow

#endif  // ONEFLOW_REGISTER_BLOB_H_
