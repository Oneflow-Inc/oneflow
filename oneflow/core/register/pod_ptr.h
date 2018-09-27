#ifndef ONEFLOW_CORE_REGISTER_POD_PTR_H_
#define ONEFLOW_CORE_REGISTER_POD_PTR_H_
#include "oneflow/core/register/pod.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

class PodPtr final {
 public:
  PodPtr(const PodProto& pod_proto, char* mem_ptr) : pod_proto_(&pod_proto), mem_ptr_(mem_ptr) {}
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

  const PodProto& pod_proto() const { return *pod_proto_; }
  bool HasField(const std::string& field_name) const;
  PodPtr Field(const std::string& field_name) const;

 private:
  template<typename T>
  void CheckDataType() {
    CHECK(pod_proto_->has_shaped_pod());
    CHECK_EQ(pod_proto_->shaped_pod().data_type(), GetDataType<T>::value);
  }
  const PodProto* pod_proto_;
  char* mem_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_POD_PTR_H_
