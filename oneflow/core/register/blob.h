#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/common/shape.h"

namespace oneflow {

class Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(void* dptr, const Shape* shape) : dptr_(dptr), shape_(shape) {}
  ~Blob() {}

  const void* dptr() const { return dptr_; }
  const Shape& shape() const { return *shape_; }

  void* mut_dptr() { return dptr_; }

 private:
  void* dptr_;
  const Shape* shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
