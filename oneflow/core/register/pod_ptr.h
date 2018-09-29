#ifndef ONEFLOW_CORE_REGISTER_POD_PTR_H_
#define ONEFLOW_CORE_REGISTER_POD_PTR_H_
#include "oneflow/core/register/pod_desc.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

class PodPtr final {
 public:
  PodPtr(const PodDesc& pod_desc, char* ptr) : pod_desc_(&pod_desc), ptr_(ptr) {}
  ~PodPtr() = default;

  template<typename T>
  const T* ShapedPtr() const;
  template<typename T>
  const T* ShapedPtr(const std::string& field_name, const T* default_ptr) const;

  template<typename T>
  T* MutShapedPtr();
  template<typename T>
  T* MutShapedPtr(const std::string& field_name, T* default_ptr);

  const PodDesc& pod_desc() const { return *pod_desc_; }
  char* ptr() const { return ptr_; }
  bool HasField(const std::string& field_name) const;
  const PodPtr Field(const std::string& field_name) const;
  PodPtr MutField(const std::string& field_name);

 private:
  template<typename T>
  void CheckDataType() {
    const auto* shaped_pod = dynamic_cast<const ShapedPodDesc*>(pod_desc_);
    CHECK_NOTNULL(shaped_pod);
    CHECK_EQ(shaped_pod->data_type(), GetDataType<T>::value);
  }

  const PodDesc* const pod_desc_;
  char* const ptr_;
};

template<typename T>
const T* PodPtr::ShapedPtr() const {
  CheckDataType<T>();
  return reinterpret_cast<const T*>(ptr_);
}

template<typename T>
const T* PodPtr::ShapedPtr(const std::string& field_name, const T* default_ptr) const {
  if (!HasField(field_name)) { return default_ptr; }
  return Field(field_name).ShapedPtr<T>();
}

template<typename T>
T* PodPtr::MutShapedPtr() {
  CheckDataType<T>();
  return reinterpret_cast<T*>(ptr_);
}

template<typename T>
T* PodPtr::MutShapedPtr(const std::string& field_name, T* default_ptr) {
  if (!HasField(field_name)) { return default_ptr; }
  return MutField(field_name).MutShapedPtr<T>();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_POD_PTR_H_
