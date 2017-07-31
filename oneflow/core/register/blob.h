#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include <cmath>
#include "oneflow/core/common/shape.h"

namespace oneflow {

class Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(void* dptr, const Shape* shape) : dptr_(dptr), shape_(shape) {}
  ~Blob() {}

  template<typename T = void>
  const T* dptr() const {
    return static_cast<const T*>(dptr_);
  }

  template<typename T = void>
  T* mut_dptr() {
    return static_cast<T*>(dptr_);
  }

  const Shape& shape() const { return *shape_; }

 private:
  void* dptr_;
  const Shape* shape_;
};

template<typename FloatingPointType = void>
bool IsFinite(const Blob* blob) {
  for (int64_t i = 0; i < blob->shape().elem_cnt(); ++i) {
    if (!std::isfinite(blob->dptr<FloatingPointType>()[i])) { return false; }
  }
  return true;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
