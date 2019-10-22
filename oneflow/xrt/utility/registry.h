#ifndef ONEFLOW_XRT_UTILITY_REGISTER_H_
#define ONEFLOW_XRT_UTILITY_REGISTER_H_

#include "glog/logging.h"
#include "oneflow/xrt/any.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {
namespace util {

class RegistryBase {
 public:
  virtual bool IsRegistered(const std::string &key) const = 0;

  RegistryBase() = default;
  virtual ~RegistryBase() = default;
};

// Global map
template <typename Factory>
class Registry : public RegistryBase {
 public:
  static Registry<Factory> *Global() {
    static auto *g_registry = new Registry<Factory>;
    return g_registry;
  }

  typedef util::Map<std::string, Any> Attribute;
  struct AttributeFactory {
    Factory factory;
    Attribute attribute;
  };

  bool IsRegistered(const std::string &key) const override {
    return factories_.count(key) > 0;
  }

  const Factory &Lookup(const std::string &key) const {
    CHECK_GT(factories_.count(key), 0)
        << "Factory (" << key << ") has not been registered.";
    return factories_.at(key).factory;
  }

  const Attribute &LookupAttr(const std::string &key) const {
    CHECK_GT(factories_.count(key), 0)
        << "Factory (" << key << ") has not been registered.";
    return factories_.at(key).attribute;
  }

  void Register(const std::string &key, Factory factory) {
    Register(key, factory, Attribute{});
  }

  void Register(const std::string &key, Factory factory,
                const Attribute &attribute) {
    if (!factories_.emplace(key, MakeFactory(factory, attribute)).second) {
      LOG(INFO) << "Factory (" << key
                << ") has been registered more than once.";
    }
  }

  virtual ~Registry() = default;

 private:
  Registry() = default;
  AttributeFactory &&MakeFactory(Factory factory, const Attribute &attribute) {
    AttributeFactory attri_factory;
    attri_factory.factory = factory;
    attri_factory.attribute = attribute;
    return std::move(attri_factory);
  }

 private:
  util::Map<std::string, AttributeFactory> factories_;
};

#define XRT_REGISTER_FACTORY(FactoryName, Factory) \
  XRT_REGISTER_FACTORY_IMPL(__COUNTER__, FactoryName, Factory)

#define XRT_REGISTER_FACTORY_IMPL(Counter, FactoryName, Factory) \
  XRT_REGISTER_FACTORY_IMPL_0(Counter, FactoryName, Factory)

#define XRT_REGISTER_FACTORY_IMPL_0(Counter, FactoryName, Factory)         \
  struct _##FactoryName##Counter {                                         \
    _##FactoryName##Counter() {                                            \
      auto _Factory = Factory;                                             \
      util::Registry<decltype(_Factory)>::Global()->Register(#FactoryName, \
                                                             _Factory);    \
    }                                                                      \
  };                                                                       \
  static _##FactoryName##Counter _registry_##FactoryName##Counter##_       \
      __attribute__((unused));

#define XRT_GLOBAL_REGISTRY(Factory) util::Registry<decltype(Factory)>::Global()

#define XRT_REGISTER_FIELD_FACTORY(Field, FactoryName, Factory) \
  XRT_REGISTER_FACTORY(FactoryName, Factory);                   \
  XRT_REGISTER_REG_MANAGER(Field, (XRT_GLOBAL_REGISTRY(Factory)));

template <typename Field>
class RegistryManager {
 public:
  static RegistryManager<Field> *Global() {
    static auto *g_registry_manager = new RegistryManager<Field>;
    return g_registry_manager;
  }

  RegistryBase *Get(const Field &field) const {
    CHECK_GT(registry_fields_.count(field), 0) << "No registry field.";
    return registry_fields_.at(field);
  }

  bool Insert(const Field &field, RegistryBase *registry) {
    return registry_fields_.emplace(field, registry).second;
  }

  virtual ~RegistryManager() = default;

 private:
  util::Map<Field, RegistryBase *> registry_fields_;
};

#define XRT_REGISTER_REG_MANAGER(Field, Registry) \
  XRT_REGISTER_REG_MANAGER_IMPL(__COUNTER__, Field, Registry)

#define XRT_REGISTER_REG_MANAGER_IMPL(Counter, Field, Registry) \
  XRT_REGISTER_REG_MANAGER_IMPL_0(Counter, Field, Registry)

#define XRT_REGISTER_REG_MANAGER_IMPL_0(Counter, Field, Registry)          \
  struct _RegistryManager_##Counter {                                      \
    _RegistryManager_##Counter() {                                         \
      using FieldType = decltype(Field);                                   \
      util::RegistryManager<FieldType>::Global()->Insert(Field, Registry); \
    }                                                                      \
  };                                                                       \
  static _RegistryManager_##Counter _registry_manager_##Counter##_         \
      __attribute__((unused));

}  // namespace util
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_UTILITY_REGISTER_H_
