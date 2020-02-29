#include "oneflow/core/extension/extension_registration.h"

namespace oneflow {

namespace extension {

namespace {
HashMap<std::string, std::vector<std::function<extension::ExtensionBase*()>>>*
MutExtensionRegistry() {
  static HashMap<std::string, std::vector<std::function<extension::ExtensionBase*()>>> registry;
  return &registry;
}

}  // namespace

Registrar::Registrar(std::string ev_name,
                     std::function<extension::ExtensionBase*()> ext_contructor) {
  auto* registry = MutExtensionRegistry();
  (*registry)[ev_name].emplace_back(std::move(ext_contructor));
}

const std::vector<std::function<extension::ExtensionBase*()>>* LookUpExtensionRegistry(
    const std::string& ev_name) {
  const auto* registry = MutExtensionRegistry();
  auto it = registry->find(ev_name);
  if (it == registry->end()) {
    return nullptr;
  } else {
    return &it->second;
  }
}

}  // namespace extension

}  // namespace oneflow
