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
#ifndef ONEFLOW_XRT_UTILITY_REGISTER_H_
#define ONEFLOW_XRT_UTILITY_REGISTER_H_

#include "glog/logging.h"
#include "oneflow/xrt/any.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {
namespace util {

template<typename Field, typename Key, typename Factory>
class FieldRegistry;

class RegistryBase {
 public:
  virtual bool IsRegistered(const Any &key) const = 0;

  typedef util::Map<std::string, Any> Attribute;
  virtual const Attribute &LookupAttr(const Any &key) const = 0;

  RegistryBase() = default;
  virtual ~RegistryBase() = default;
};

// Global factory map
template<typename Key, typename Factory>
class Registry : public RegistryBase {
 public:
  static Registry<Key, Factory> *Global() {
    static auto *g_registry = new Registry<Key, Factory>;
    return g_registry;
  }

  struct AttributeFactory {
    Factory factory;
    Attribute attribute;
  };

  bool IsRegistered(const Any &key) const override {
    return factories_.count(any_cast<Key>(key)) > 0;
  }

  const Factory &Lookup(const Any &key) const { return Lookup(any_cast<Key>(key)); }

  const Factory &Lookup(const Key &key) const {
    CHECK_GT(factories_.count(key), 0) << "Factory (" << key << ") has not been registered.";
    return factories_.at(key).factory;
  }

  const Attribute &LookupAttr(const Any &key) const { return LookupAttr(any_cast<Key>(key)); }

  const Attribute &LookupAttr(const Key &key) const {
    CHECK_GT(factories_.count(key), 0) << "Factory (" << key << ") has not been registered.";
    return factories_.at(key).attribute;
  }

  void Register(const Key &key, Factory factory) { Register(key, factory, Attribute{}); }

  void Register(const Key &key, Factory factory, const Attribute &attribute) {
    if (!factories_.emplace(key, MakeFactory(factory, attribute)).second) {
      LOG(INFO) << "Factory (" << key << ") has been registered more than once.";
    }
  }

  virtual ~Registry() = default;

  template<typename FieldT, typename KeyT, typename FactoryT>
  friend class FieldRegistry;

 protected:
  Registry() = default;
  AttributeFactory MakeFactory(Factory factory, const Attribute &attribute) {
    AttributeFactory attri_factory;
    attri_factory.factory = factory;
    attri_factory.attribute = attribute;
    return std::move(attri_factory);
  }

 private:
  util::Map<Key, AttributeFactory> factories_;
};

#define XRT_REGISTER_FACTORY(FactoryName, Factory) \
  XRT_REGISTER_FACTORY_IMPL(__COUNTER__, FactoryName, Factory)

#define XRT_REGISTER_FACTORY_IMPL(Counter, FactoryName, Factory) \
  XRT_REGISTER_FACTORY_IMPL_0(Counter, FactoryName, Factory)

#define XRT_REGISTER_FACTORY_IMPL_0(Counter, FactoryName, Factory)                              \
  namespace {                                                                                   \
  struct _XrtRegistry_##Counter {                                                               \
    _XrtRegistry_##Counter() {                                                                  \
      util::Registry<decltype(FactoryName), decltype(Factory)>::Global()->Register(FactoryName, \
                                                                                   Factory);    \
    }                                                                                           \
  };                                                                                            \
  static _XrtRegistry_##Counter _xrt_registry_##Counter##_ __attribute__((unused));             \
  }  // namespace

template<typename Field>
class RegistryManager {
 public:
  static RegistryManager<Field> *Global() {
    static auto *g_registry_manager = new RegistryManager<Field>;
    return g_registry_manager;
  }

  RegistryBase *GetRegistry(const Field &field) const {
    CHECK_GT(registry_fields_.count(field), 0) << "No registry field.";
    return registry_fields_.at(field);
  }

  bool Insert(const Field &field, RegistryBase *registry) {
    return registry_fields_.emplace(field, registry).second;
  }

  bool HasRegistry(const Field &field) const { return registry_fields_.count(field) > 0; }

  virtual ~RegistryManager() = default;

 private:
  util::Map<Field, RegistryBase *> registry_fields_;
};

template<typename Field, typename Key, typename Factory>
class FieldRegistry {
 public:
  static FieldRegistry<Field, Key, Factory> *Global() {
    static auto *g_field_registry = new FieldRegistry<Field, Key, Factory>();
    return g_field_registry;
  }

  Registry<Key, Factory> *Get(const Field &field) {
    auto it = field_factories_.find(field);
    if (it == field_factories_.end()) {
      it = field_factories_.emplace(field, Registry<Key, Factory>()).first;
      registry_manager_->Insert(field, &(it->second));
    }
    return &(it->second);
  }

  virtual ~FieldRegistry() = default;

 protected:
  FieldRegistry() : registry_manager_(RegistryManager<Field>::Global()) {}

 private:
  util::Map<Field, Registry<Key, Factory>> field_factories_;
  RegistryManager<Field> *registry_manager_;
};

}  // namespace util
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_UTILITY_REGISTER_H_
