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
  const T* ShapedPtr(FieldKey field_key, const T* default_ptr) const;

  template<typename T>
  T* MutShapedPtr();
  template<typename T>
  T* MutShapedPtr(FieldKey field_key, T* default_ptr);

  const PodDesc& pod_desc() const { return *pod_desc_; }
  char* ptr() const { return ptr_; }
  bool HasField(FieldKey field_key) const { return HasField(NewFieldId(field_key)); }
  const PodPtr Field(FieldKey field_key) const { return Field(NewFieldId(field_key)); }
  PodPtr MutField(FieldKey field_key) { return MutField(NewFieldId(field_key)); }

  bool HasField(const FieldId& field_id) const;
  const PodPtr Field(const FieldId& field_id) const;
  PodPtr MutField(const FieldId& field_id);

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
const T* PodPtr::ShapedPtr(FieldKey field_key, const T* default_ptr) const {
  if (!HasField(field_key)) { return default_ptr; }
  return Field(field_key).ShapedPtr<T>();
}

template<typename T>
T* PodPtr::MutShapedPtr(FieldKey field_key, T* default_ptr) {
  if (!HasField(field_key)) { return default_ptr; }
  return MutField(field_key).MutShapedPtr<T>();
}

template<typename T>
const T* PodPtr::ShapedPtr() const {
  CheckDataType<T>();
  return reinterpret_cast<const T*>(ptr_);
}

template<typename T>
T* PodPtr::MutShapedPtr() {
  CheckDataType<T>();
  return reinterpret_cast<T*>(ptr_);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_POD_PTR_H_
