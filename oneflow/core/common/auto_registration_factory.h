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
#ifndef ONEFLOW_CORE_COMMON_AUTO_REGISTRATION_FACTORY_H_
#define ONEFLOW_CORE_COMMON_AUTO_REGISTRATION_FACTORY_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename Base, typename... Args>
struct AutoRegistrationFactory {
 public:
  template<typename Derived>
  struct RawRegisterType {
    RawRegisterType(int32_t k) {
      CHECK((AutoRegistrationFactory<Base, Args...>::Get()
                 .creators_.emplace(k, [](Args&&...) { return new Derived; })
                 .second))
          << k;
    }
  };

  struct CreatorRegisterType {
    CreatorRegisterType(int32_t k, std::function<Base*(Args&&...)> v) {
      CHECK((AutoRegistrationFactory<Base, Args...>::Get().creators_.emplace(k, v).second)) << k;
    }
  };

  Base* New(int32_t k, Args&&... args) {
    auto creators_it = creators_.find(k);
    CHECK(creators_it != creators_.end()) << "Unregistered: " << k;
    return creators_it->second(std::forward<Args>(args)...);
  }

  bool IsClassRegistered(int32_t k, Args&&... args) { return creators_.find(k) != creators_.end(); }

  static AutoRegistrationFactory<Base, Args...>& Get() {
    static AutoRegistrationFactory<Base, Args...> obj;
    return obj;
  }

 private:
  HashMap<int32_t, std::function<Base*(Args&&...)>> creators_;
};

#define REGISTER_VAR_NAME OF_PP_CAT(g_registry_var, __COUNTER__)

#define REGISTER_CLASS(k, Base, Derived) \
  static AutoRegistrationFactory<Base>::RawRegisterType<Derived> REGISTER_VAR_NAME(k)
#define REGISTER_CLASS_WITH_ARGS(k, Base, Derived, ...) \
  static AutoRegistrationFactory<Base, __VA_ARGS__>::RawRegisterType<Derived> REGISTER_VAR_NAME(k)
#define REGISTER_CLASS_CREATOR(k, Base, f, ...) \
  static AutoRegistrationFactory<Base, ##__VA_ARGS__>::CreatorRegisterType REGISTER_VAR_NAME(k, f)

template<typename Base, typename... Args>
inline Base* NewObj(int32_t k, Args&&... args) {
  return AutoRegistrationFactory<Base, Args...>::Get().New(k, std::forward<Args>(args)...);
}

template<typename Base, typename... Args>
inline bool IsClassRegistered(int32_t k, Args&&... args) {
  return AutoRegistrationFactory<Base, Args...>::Get().IsClassRegistered(
      k, std::forward<Args>(args)...);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_AUTO_REGISTRATION_FACTORY_H_
