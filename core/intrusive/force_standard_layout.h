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
#ifndef ONEFLOW_CORE_INTRUSIVE_FORCE_STANDARD_LAYOUT_H_
#define ONEFLOW_CORE_INTRUSIVE_FORCE_STANDARD_LAYOUT_H_

namespace oneflow {
namespace intrusive {

template<typename T>
class ForceStandardLayout final {
 public:
  ForceStandardLayout() { new (&object_) T(); }
  template<typename Arg, typename = typename std::enable_if<!std::is_same<
                             ForceStandardLayout, typename std::decay<Arg>::type>::value>::type>
  explicit ForceStandardLayout(Arg&& arg) {
    new (&object_) T(std::forward<Arg>(arg));
  }
  template<typename Arg0, typename Arg1, typename... Args>
  ForceStandardLayout(Arg0&& arg0, Arg1&& arg1, Args&&... args) {
    new (&object_)
        T(std::forward<Arg0>(arg0), std::forward<Arg1>(arg1), std::forward<Args>(args)...);
  }

  ~ForceStandardLayout() { Mutable()->~T(); }

  ForceStandardLayout(const ForceStandardLayout& other) { new (&object_) T(other.Get()); }
  ForceStandardLayout(ForceStandardLayout&& other) {
    new (&object_) T(std::move(*other.Mutable()));
  }

  ForceStandardLayout& operator=(const ForceStandardLayout& other) {
    *Mutable() = other.Get();
    return *this;
  }
  ForceStandardLayout& operator=(ForceStandardLayout&& other) {
    *Mutable() = std::move(*other.Mutable());
    return *this;
  }

  const T& Get() const {
    const auto* __attribute__((__may_alias__)) ptr = reinterpret_cast<const T*>(&object_[0]);
    return *ptr;
  }

  T* Mutable() {
    auto* __attribute__((__may_alias__)) ptr = reinterpret_cast<T*>(&object_[0]);
    return ptr;
  }

 private:
  alignas(T) char object_[sizeof(T)];
};

}  // namespace intrusive
}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_FORCE_STANDARD_LAYOUT_H_
