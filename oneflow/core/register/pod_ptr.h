#ifndef ONEFLOW_CORE_REGISTER_POD_PTR_H_
#define ONEFLOW_CORE_REGISTER_POD_PTR_H_
#include "oneflow/core/register/pod_desc.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

class PodPtr final {
 public:
  PodPtr(const PodDesc& pod_desc, char* mem_ptr) : pod_desc_(&pod_desc), mem_ptr_(mem_ptr) {}
  ~PodPtr() = default;

  template<typename T>
  const T* Get() const {
    CheckDataType<T>();
    return static_cast<T*>(mem_ptr_);
  }

  template<typename T>
  T* Mut() {
    CheckDataType<T>();
    return static_cast<T*>(mem_ptr_);
  }

  const PodDesc& pod_desc() const { return *pod_desc_; }
  char* mem_ptr() const { return mem_ptr_; }
  bool HasField(const std::string& field_name) const;
  PodPtr Field(const std::string& field_name) const;

 private:
  template<typename T>
  void CheckDataType() {
    const auto* shaped_pod = dynamic_cast<const ShapedPodDesc*>(pod_desc_);
    CHECK_NOTNULL(shaped_pod);
    CHECK_EQ(shaped_pod->data_type(), GetDataType<T>::value);
  }
  const PodDesc* pod_desc_;
  char* mem_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_POD_PTR_H_
