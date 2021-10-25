/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_INTRUSIVE_REF_H_
#define ONEFLOW_CORE_INTRUSIVE_REF_H_

#include <atomic>
#include <glog/logging.h>

namespace oneflow {

namespace intrusive {

class Ref {
 public:
  Ref() : ref_cnt_() {}

  using RefCntType = int32_t;

  RefCntType ref_cnt() const { return ref_cnt_; }

  template<typename T>
  static void NewAndInitRef(T** ptr) {
    *ptr = new T();
    (*ptr)->mut_intrusive_ref()->InitRefCount();
    IncreaseRef(*ptr);
  }
  template<typename T>
  static void IncreaseRef(T* ptr) {
    ptr->mut_intrusive_ref()->IncreaseRefCount();
  }
  template<typename T>
  static void DecreaseRef(T* ptr) {
    CHECK_NOTNULL(ptr);
    RefCntType ref_cnt = ptr->mut_intrusive_ref()->DecreaseRefCount();
    if (ref_cnt > 0) { return; }
    ptr->__Delete__();
    delete ptr;
  }

 private:
  void InitRefCount() { ref_cnt_ = 0; }
  void IncreaseRefCount() { ref_cnt_++; }
  RefCntType DecreaseRefCount() { return --ref_cnt_; }

  std::atomic<RefCntType> ref_cnt_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_REF_H_
