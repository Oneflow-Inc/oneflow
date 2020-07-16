#ifndef ONEFLOW_CORE_VM_OBJECT_WRAPPER_H_
#define ONEFLOW_CORE_VM_OBJECT_WRAPPER_H_

#include <memory>
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace vm {

template<typename T>
class ObjectWrapper final : public Object {
 public:
  explicit ObjectWrapper(const std::shared_ptr<T>& data) : data_(data) {}

  ~ObjectWrapper() = default;

  const T& operator*() const { return *data_; }
  T& operator*() { return *data_; }
  const T* operator->() const { return data_.get(); }
  T* operator->() { return data_.get(); }

  const std::shared_ptr<T>& GetPtr() const { return data_; }
  const T& Get() const { return *data_; }
  T* Mutable() { return data_.get(); }

 private:
  std::shared_ptr<T> data_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_OBJECT_WRAPPER_H_
