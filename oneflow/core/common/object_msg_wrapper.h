#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_WRAPPER_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_WRAPPER_H_

#include <glog/logging.h>
#include "oneflow/core/common/object_msg_core.h"

namespace oneflow {

// clang-format off
template<typename CppObjectT>
BEGIN_OBJECT_MSG(Wrapper4CppObject);
 public:
  template<typename NewCppObjectT>
  void __Init__(const NewCppObjectT& NewCppObject) {
    set_obj(NewCppObject(mut_allocator(), mutable_obj_size()));
    CHECK_GE(obj_size(), 0);
  }
  void __Init__() {
    __Init__([](ObjectMsgAllocator* allocator, int32_t* obj_size){
        *obj_size = sizeof(CppObjectT);
        char* mem_ptr = allocator->Allocate(sizeof(CppObjectT));
        return new (mem_ptr) CppObjectT();
    });
  }
  void __Delete__() {
    if (obj_size() == 0) { return; }
    mut_allocator()->Deallocate(reinterpret_cast<char*>(mutable_obj()), obj_size());
  }

  const CppObjectT& operator*() const { return obj(); }
  CppObjectT& operator*() { return *mutable_obj(); }

  const CppObjectT* operator->() const { return &obj(); }
  CppObjectT* operator->() { return mutable_obj(); }

  const CppObjectT& Get() const { return obj(); }
  CppObjectT* Mutable() { return mutable_obj(); }

  OBJECT_MSG_DEFINE_RAW_PTR(CppObjectT, obj);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, obj_size);
END_OBJECT_MSG(Wrapper4CppObject);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_WRAPPER_H_
