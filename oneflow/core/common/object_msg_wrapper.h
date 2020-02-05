#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_WRAPPER_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_WRAPPER_H_

#include <glog/logging.h>
#include "oneflow/core/common/object_msg_core.h"

namespace oneflow {

// clang-format off
template<typename CppObjectT>
BEGIN_OBJECT_MSG(Wrapper4CppObject);
 public:
  template<typename NewQuerierT>
  void __Init__(const NewQuerierT& NewQuerier) {
    set_obj(NewQuerier(mut_allocator(), mutable_obj_size()));
    CHECK_GE(obj_size(), 0);
  }
  void __Delete__() {
    if (obj_size() == 0) { return; }
    mut_allocator()->Deallocate(reinterpret_cast<char*>(mutable_obj()), obj_size());
  }

  const CppObjectT& operator*() const { return *obj(); }
  CppObjectT& operator*() { return *mutable_obj(); }

  const CppObjectT* operator->() const { return obj(); }
  CppObjectT* operator->() { return mutable_obj(); }

  const CppObjectT& Get() const { return *obj(); }
  CppObjectT* Mutable() { return mutable_obj(); }

  OBJECT_MSG_DEFINE_RAW_PTR(CppObjectT*, obj);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, obj_size);
END_OBJECT_MSG(Wrapper4CppObject);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_WRAPPER_H_
