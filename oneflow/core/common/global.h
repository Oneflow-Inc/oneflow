#ifndef ONEFLOW_CORE_COMMON_GLOBAL_H_
#define ONEFLOW_CORE_COMMON_GLOBAL_H_

#include <glog/logging.h>
#include "oneflow/core/common/maybe.h"

namespace oneflow {

template<typename T, typename Kind = void>
class Global final {
 public:
  static T* Get() { return *GetPPtr(); }
  static void SetAllocated(T* val) { *GetPPtr() = val; }
  template<typename... Args>
  static void New(Args&&... args) {
    CHECK(Get() == nullptr);
    LOG(INFO) << "NewGlobal " << typeid(T).name();
    *GetPPtr() = new T(std::forward<Args>(args)...);
  }
  static void Delete() {
    if (Get() != nullptr) {
      LOG(INFO) << "DeleteGlobal " << typeid(T).name();
      delete Get();
      *GetPPtr() = nullptr;
    }
  }

 private:
  static T** GetPPtr() {
    CheckKind();
    static T* ptr = nullptr;
    return &ptr;
  }
  static void CheckKind() {
    if (!std::is_same<Kind, void>::value) {
      CHECK(Global<T>::Get() == nullptr)
          << typeid(Global<T>).name() << " are disable for avoiding misuse";
    }
  }
};

template<typename T, typename... Kind>
Maybe<T*> GlobalMaybe() {
  CHECK_NOTNULL_OR_RETURN((Global<T, Kind...>::Get())) << " typeid: " << typeid(T).name();
  return Global<T, Kind...>::Get();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_GLOBAL_H_
