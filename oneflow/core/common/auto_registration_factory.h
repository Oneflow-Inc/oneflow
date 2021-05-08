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

// 工厂模式+单例模式
// 注册工厂类模板
// 模板参数Key为派生类关键字，Base为基类
template<typename Key, typename Base, typename... Args>
struct AutoRegistrationFactory {
 public:
  using Creator = std::function<Base*(Args&&...)>;
  template<typename Derived>
  struct RawRegisterType {
    RawRegisterType(Key k) {
      // 注册Derived类，即保存实例化Derived的函数
      CHECK((AutoRegistrationFactory<Key, Base, Args...>::Get()
                 .mutable_creators()
                 ->emplace(k, [](Args&&...) { return new Derived; })
                 .second))
          << k;
    }
  };

  struct CreatorRegisterType {
    CreatorRegisterType(Key k, Creator v) {
      // 注册Derived类，用户自定义实例化Derived的函数
      CHECK((AutoRegistrationFactory<Key, Base, Args...>::Get()
                 .mutable_creators()
                 ->emplace(k, v)
                 .second))
          << k;
    }
  };

  // 若Derive已注册，创建实例
  Base* New(Key k, Args&&... args) const {
    auto creators_it = creators().find(k);
    CHECK(creators_it != creators().end()) << "Unregistered: " << k;
    return creators_it->second(std::forward<Args>(args)...);
  }

  bool IsClassRegistered(Key k, Args&&... args) const {
    return creators().find(k) != creators().end();
  }

  static AutoRegistrationFactory<Key, Base, Args...>& Get() {
    static AutoRegistrationFactory<Key, Base, Args...> obj;
    return obj;
  }

 private:
  // 保存<Derived Key，实例化对应Derived的函数>的HashMap
  std::unique_ptr<HashMap<Key, Creator>> creators_;

  bool has_creators() const { return creators_.get() != nullptr; }

  const HashMap<Key, Creator>& creators() const {
    CHECK(has_creators()) << "Unregistered key type: " << typeid(Key).name();
    return *creators_.get();
  }

  HashMap<Key, Creator>* mutable_creators() {
    if (!creators_) { creators_.reset(new HashMap<Key, Creator>); }
    return creators_.get();
  }
};

#define REGISTER_VAR_NAME OF_PP_CAT(g_registry_var, __COUNTER__)

// REGISTER_CLASS，用于注册Derived类
// 使用全局变量（RawRegisterType）的构造函数，从而在main之前运行
// 示例：oneflow/core/actor/acc_compute_actor.cpp.i，宏展开前后：
// REGISTER_ACTOR(TaskType::kAcc, AccCompActor);
// static AutoRegistrationFactory<int32_t, Actor>::RawRegisterType<AccCompActor> g_registry_var1(TaskType::kAcc);
#define REGISTER_CLASS(Key, k, Base, Derived) \
  static AutoRegistrationFactory<Key, Base>::RawRegisterType<Derived> REGISTER_VAR_NAME(k)
#define REGISTER_CLASS_WITH_ARGS(Key, k, Base, Derived, ...)                       \
  static AutoRegistrationFactory<Key, Base, __VA_ARGS__>::RawRegisterType<Derived> \
      REGISTER_VAR_NAME(k)
// @PROB(shiyongtao): 未提供自定义CREATOR对应的Destroy，可能存在内存泄漏等风险
#define REGISTER_CLASS_CREATOR(Key, k, Base, f, ...)                                               \
  static AutoRegistrationFactory<Key, Base, ##__VA_ARGS__>::CreatorRegisterType REGISTER_VAR_NAME( \
      k, f)

// 实例化已注册的类
template<typename Key, typename Base, typename... Args>
inline Base* NewObj(Key k, Args&&... args) {
  return AutoRegistrationFactory<Key, Base, Args...>::Get().New(k, std::forward<Args>(args)...);
}

template<typename Key, typename Base, typename... Args>
inline bool IsClassRegistered(Key k, Args&&... args) {
  return AutoRegistrationFactory<Key, Base, Args...>::Get().IsClassRegistered(
      k, std::forward<Args>(args)...);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_AUTO_REGISTRATION_FACTORY_H_
