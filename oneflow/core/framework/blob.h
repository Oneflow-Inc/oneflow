#ifndef ONEFLOW_CORE_FRAMEWORK_BLOB_H_
#define ONEFLOW_CORE_FRAMEWORK_BLOB_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/blob_def.h"

namespace oneflow {

class Shape;

namespace user_op {

class Blob final {
 public:
  Blob(const BlobDef&, char*);
  Blob(const Shape&, DataType, char*);
  ~Blob() = default;

  Blob(const Blob&);

  const Shape& shape() const { return def_.shape(); }
  DataType data_type() const { return def_.data_type(); }

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
            && def_.data_type() != DataType::kChar && def_.data_type() != GetDataType<T>::value))
        << def_.data_type() << " " << GetDataType<T>::value;
  }

  BlobDef def_;
  void* dptr_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_BLOB_H_
